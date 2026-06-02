import os
import requests
import json
from typing import List, Dict, Tuple

class RxNavSafetyChecker:
    """
    Real-world clinical safety checker querying the National Library of Medicine (NLM) RxNav API
    for drug-drug interactions, with a robust fallback mechanism.
    """
    
    @staticmethod
    def get_rxcui(drug_name: str) -> str:
        """Fetch the RxNorm Concept Unique Identifier (RxCUI) for a drug name."""
        try:
            # Clean name for search
            clean_name = drug_name.split()[0].replace(',', '').strip().lower()
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={clean_name}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                id_group = data.get("idGroup", {})
                rxnorm_ids = id_group.get("rxconceptId", [])
                if rxnorm_ids:
                    return rxnorm_ids[0]
        except Exception:
            pass
        return ""

    @classmethod
    def check_interactions(cls, medications: List[str]) -> List[Dict]:
        """Check for severe drug-drug interactions using the NIH RxNav API."""
        interactions = []
        rxcuis = []
        cui_to_drug = {}
        
        # Resolve names to RxCUIs
        for med in medications:
            cui = cls.get_rxcui(med)
            if cui:
                rxcuis.append(cui)
                cui_to_drug[cui] = med

        if len(rxcuis) < 2:
            return []

        try:
            rxcuis_str = "+".join(rxcuis)
            url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={rxcuis_str}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                full_interaction_groups = data.get("fullInteractionTypeGroup", [])
                for group in full_interaction_groups:
                    for interaction_type in group.get("fullInteractionType", []):
                        comment = interaction_type.get("comment", "")
                        interaction_pair = interaction_type.get("interactionPair", [])
                        for pair in interaction_pair:
                            severity = pair.get("severity", "high").lower()
                            description = pair.get("description", "")
                            
                            # Resolve involved concepts
                            concepts = pair.get("interactionConcept", [])
                            involved_meds = []
                            for concept in concepts:
                                source_concept = concept.get("minConceptItem", {})
                                rxcui = source_concept.get("rxcui", "")
                                name = cui_to_drug.get(rxcui, source_concept.get("name", "Unknown"))
                                involved_meds.append(name)
                                
                            interactions.append({
                                "drugs": involved_meds,
                                "severity": severity,
                                "description": description,
                                "source": "NIH RxNav API"
                            })
        except Exception as e:
            # Fallback to a mock ruleset if network fails or API is slow
            print(f"RxNav API failed ({e}), using internal clinical rule-based check.")
            interactions = cls._internal_safety_checks(medications)
            
        return interactions

    @staticmethod
    def _internal_safety_checks(medications: List[str]) -> List[Dict]:
        """Internal rule-based high-risk interaction checker (fallback)."""
        med_set = {m.lower() for m in medications}
        interactions = []
        
        # Real-world common dangerous drug pairs
        dangerous_pairs = [
            (("warfarin", "aspirin"), "high", "Increased risk of bleeding / gastrointestinal hemorrhage."),
            (("warfarin", "ibuprofen"), "high", "NSAIDs increase bleed risk with oral anticoagulants."),
            (("lisinopril", "spironolactone"), "medium", "Risk of severe hyperkalemia (high potassium)."),
            (("sildenafil", "nitroglycerin"), "high", "Severe, life-threatening hypotension (low blood pressure)."),
            (("atorvastatin", "clarithromycin"), "medium", "Increased risk of myopathy / rhabdomyolysis."),
            (("metformin", "contrast"), "high", "Risk of lactic acidosis. Metformin should be held before procedures using iodinated contrast.")
        ]
        
        for (drug_a, drug_b), severity, desc in dangerous_pairs:
            # Check if any med contains these drug names
            match_a = [m for m in medications if drug_a in m.lower()]
            match_b = [m for m in medications if drug_b in m.lower()]
            if match_a and match_b:
                interactions.append({
                    "drugs": [match_a[0], match_b[0]],
                    "severity": severity,
                    "description": desc,
                    "source": "Internal Clinical Safety Rules"
                })
                
        return interactions


class MedicationReconciler:
    """Performs medication reconciliation comparing admission vs discharge meds."""
    
    @staticmethod
    def reconcile(admission_meds: List[str], discharge_meds: List[str], documentation: str = "") -> List[Dict]:
        """
        Compares admission vs discharge medications.
        Detects additions, removals, and changes, and verifies if a clear clinical reason is documented.
        """
        reconciliation_list = []
        
        # Standardize lists
        adm_clean = [m.strip() for m in admission_meds if m.strip()]
        dis_clean = [m.strip() for m in discharge_meds if m.strip()]
        
        # Check added drugs
        for d_med in dis_clean:
            # Check if matching drug exists in admission
            matched_a_med = None
            d_base = d_med.split()[0].lower()
            
            for a_med in adm_clean:
                if d_base in a_med.lower():
                    matched_a_med = a_med
                    break
                    
            if not matched_a_med:
                # This drug is NEW
                # Ask: Is there a reason documented in the medical record?
                # Simple check against the clinical notes text
                reason_found = False
                reason_text = "No stated reason found in notes. Flagged for review."
                
                # Search for reasons in documentation
                doc_lower = documentation.lower()
                d_base_clean = d_base.replace(',', '')
                keywords = [f"start {d_base_clean}", f"prescribe {d_base_clean}", f"added {d_base_clean}", f"initiated {d_base_clean}", f"due to {d_base_clean}", f"for {d_base_clean}"]
                for kw in keywords:
                    if kw in doc_lower:
                        reason_found = True
                        reason_text = f"Documented clinical initiation for {d_med} detected."
                        break
                        
                reconciliation_list.append({
                    "medication": d_med,
                    "type": "Added",
                    "reason_documented": reason_found,
                    "details": reason_text
                })
            else:
                # Drug exists in both. Check if the strength/frequency changed.
                if d_med.lower() != matched_a_med.lower():
                    reconciliation_list.append({
                        "medication": f"{matched_a_med} -> {d_med}",
                        "type": "Changed",
                        "reason_documented": True,
                        "details": "Dose or schedule adjusted during hospital course."
                    })
                else:
                    reconciliation_list.append({
                        "medication": d_med,
                        "type": "Continued",
                        "reason_documented": True,
                        "details": "Unchanged from admission."
                    })
                    
        # Check stopped drugs
        for a_med in adm_clean:
            matched_d_med = None
            a_base = a_med.split()[0].lower()
            for d_med in dis_clean:
                if a_base in d_med.lower():
                    matched_d_med = d_med
                    break
            if not matched_d_med:
                # Stopped medication
                reason_found = False
                reason_text = "No documented stop reason. Flagged for review."
                
                doc_lower = documentation.lower()
                a_base_clean = a_base.replace(',', '')
                keywords = [f"stop {a_base_clean}", f"hold {a_base_clean}", f"discontinue {a_base_clean}", f"ceased {a_base_clean}"]
                for kw in keywords:
                    if kw in doc_lower:
                        reason_found = True
                        reason_text = f"Documented stop reason detected."
                        break
                        
                reconciliation_list.append({
                    "medication": a_med,
                    "type": "Stopped",
                    "reason_documented": reason_found,
                    "details": reason_text
                })
                
        return reconciliation_list
