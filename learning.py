import os
import json
from typing import Dict, List, Tuple

class LearningLoop:
    """
    Part 2: Learning from Doctor Edits.
    Handles Levenshtein distance calculations, structured correction memory, 
    and prompt injection logic to improve future agent runs.
    """
    
    MEMORY_FILE = "doctor_correction_memory.json"
    
    @staticmethod
    def calculate_levenshtein(s1: str, s2: str) -> int:
        """Calculate raw Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return LearningLoop.calculate_levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    @classmethod
    def get_accuracy_signal(cls, draft: str, edited: str) -> float:
        """
        Calculates normalized edit similarity score.
        1.0 means perfect match (no editing), 0.0 means completely different.
        """
        if not draft and not edited:
            return 1.0
        max_len = max(len(draft), len(edited))
        if max_len == 0:
            return 1.0
        distance = cls.calculate_levenshtein(draft, edited)
        # Normalize between 0.0 and 1.0
        score = 1.0 - (distance / max_len)
        return max(0.0, score)

    @classmethod
    def save_correction(cls, section_name: str, original_content: str, edited_content: str):
        """Save a clinician correction into structured correction memory."""
        memory = cls.load_memory()
        
        # We save it as a correction rule or a list of style preferences
        if section_name not in memory:
            memory[section_name] = []
            
        # Avoid saving duplicate identical corrections
        exists = False
        for record in memory[section_name]:
            if record["original"] == original_content and record["edited"] == edited_content:
                exists = True
                break
                
        if not exists:
            memory[section_name].append({
                "original": original_content,
                "edited": edited_content,
                "timestamp": os.path.getmtime(cls.MEMORY_FILE) if os.path.exists(cls.MEMORY_FILE) else 0.0
            })
            
        with open(cls.MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=4)

    @classmethod
    def load_memory(cls) -> Dict[str, List[Dict]]:
        """Load correction memory from file."""
        if not os.path.exists(cls.MEMORY_FILE):
            return {}
        try:
            with open(cls.MEMORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def clear_memory(cls):
        """Clear all learned memory (reset loop)."""
        if os.path.exists(cls.MEMORY_FILE):
            os.remove(cls.MEMORY_FILE)

    @classmethod
    def get_prompt_instructions(cls) -> str:
        """
        Construct prompt instructions injected into the agent prompt
        based on past clinician edits.
        """
        memory = cls.load_memory()
        if not memory:
            return ""
            
        instructions = "\n\n=== CLINICIAN PREFERENCE & EDIT HISTORY ===\n"
        instructions += "The clinician has previously edited drafts you generated. Study these changes and strictly follow their preferences:\n"
        
        for section, items in memory.items():
            instructions += f"\n* Section: {section}\n"
            for idx, item in enumerate(items[-3:]): # Use the 3 most recent corrections to avoid bloat
                instructions += f"  - Previous Draft: \"{item['original'].strip()}\"\n"
                instructions += f"  - Clinician's Approved Version: \"{item['edited'].strip()}\"\n"
                instructions += f"  - Policy: Adapt your output style and level of detail to mimic the clinician's approved version.\n"
        
        instructions += "\n===========================================\n"
        return instructions


class SimulatedClinician:
    """
    Simulated doctor reviewer that applies a consistent policy (e.g. style, format)
    to mock real clinician edits. This simulates the feedback loop for training/testing.
    """
    
    @staticmethod
    def review_draft(draft_data: Dict[str, str]) -> Dict[str, str]:
        """
        Applies consistent clinician policies to the draft:
        1. Demographics: Prefers a clean, standardized list instead of prose.
        2. Allergies: Must list in UPPERCASE for safety visual alert.
        3. Pending Results: Prefers bolding pending test names.
        4. Hospital Course: Prefers bullet points over long dense paragraphs.
        """
        edited_data = draft_data.copy()
        
        # Policy 1: Demographics list style
        if "demographics" in edited_data:
            demo = edited_data["demographics"]
            if "Patient Name:" not in demo and ":" in demo:
                # Format to a standard format if not already
                pass
                
        # Policy 2: Uppercase allergies
        if "allergies" in edited_data:
            allergies = edited_data["allergies"]
            # Convert text like "Penicillin" to "PENICILLIN"
            for word in ["penicillin", "sulfa", "aspirin", "peanuts", "latex", "no known drug allergies", "nkda"]:
                if word in allergies.lower():
                    allergies = allergies.replace(word, word.upper())
                    allergies = allergies.replace(word.capitalize(), word.upper())
            edited_data["allergies"] = allergies
            
        # Policy 3: Bulleted Hospital Course
        if "hospital_course" in edited_data:
            course = edited_data["hospital_course"]
            # If the course is one single massive paragraph, the simulated clinician breaks it into clean bullet points
            if len(course.split('\n')) < 3 and len(course) > 150:
                sentences = course.split('. ')
                course_bullets = []
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        if not s_clean.endswith('.'):
                            s_clean += '.'
                        course_bullets.append(f"* {s_clean}")
                edited_data["hospital_course"] = "\n".join(course_bullets)
                
        # Policy 4: Bolding pending results
        if "pending_results" in edited_data:
            pending = edited_data["pending_results"]
            # Highlight pending things
            words = ["pending", "awaited", "scheduled"]
            lines = pending.split('\n')
            for i, line in enumerate(lines):
                for w in words:
                    if w in line.lower() and "**" not in line:
                        lines[i] = f"**{line}**"
            edited_data["pending_results"] = "\n".join(lines)
            
        return edited_data
