from src.external.KARI.kari import SingleRule
import re 
import os 
import statistics
from itertools import combinations
from collections import defaultdict, Counter

import re
import os
import statistics
from collections import defaultdict, Counter
import numpy as np

# --- Configuration ---
# Input file containing the sequences with durations
# SEQUENCE_CORPUS_FILE = '/content/rarp_jigsaw_triplets.txt'
SEQUENCE_CORPUS_FILE= "/content/all_triplets.txt"
# Output file where the learned hierarchical grammar will be saved
OUTPUT_GRAMMAR_FILE = 'hierarchical_surgical_grammar_v3_rarp_jigsaw_cholec_chiral.pcfg'
# --- KARI-inspired Hyperparameters ---
# Minimum number of times an n-gram must appear to be considered a composite action
MIN_NGRAM_FREQUENCY = 5
# Maximum length of n-grams to search for (e.g., pairs, triplets, quadruplets)
MAX_NGRAM_SIZE = 4
# The order of the Markov model for transitions (e.g., 2 means P(St | St-1, St-2))
MARKOV_ORDER = 2

CHIRALITY_VERB_PAIRS = {
    "pick": "drop", "push": "pull", "tie": "untie", "tighten": "loosen", "grasp": "retract",
}
class HierarchicalGrammarLearner:
    """
    Learns a hierarchical PCFG and robust duration model inspired by the KARI methodology.
    
    Improvements based on user feedback:
    - Robust Naming: Uses unique IDs for composites (e.g., COMPOSITE_1) to avoid collisions.
    - Higher-Order Transitions: Implements an N-th order Markov model for grammar rules.
    - Robust Statistics: Calculates Median, MAD, and uses sample standard deviation.
    - Improved Replacement: Uses a left-to-right maximal match for composite replacement.
    - Probability Validation: Asserts that transition probabilities for each state sum to 1.
    """
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.duration_data = defaultdict(list)
        self.state_occurrence_count = Counter()
        self.composite_definitions = {}  # Stores definitions and aliases of composite actions
        self.start_symbol = 'S'
        self.pad_symbol = ('<PAD>',) # Padding for higher-order model history
        self.chirality_map_full = {**CHIRALITY_VERB_PAIRS}
        for k, v in CHIRALITY_VERB_PAIRS.items():
            self.chirality_map_full[v] = k

    def _get_verb_from_state(self, state_tuple):
        """Extracts the verb from a simple atomic action tuple."""
        if not state_tuple or len(state_tuple) != 1 or state_tuple[0].startswith("COMPOSITE_"):
            return None
        # Assumes format like 'verb-target-instrument'
        return state_tuple[0].split(',')[0]
    

    def _get_state_name(self, state_tuple):
        """Creates a safe, readable name for a state tuple."""
        # Handle composite states which are already named strings (now unique IDs)
        if len(state_tuple) == 1 and state_tuple[0].startswith("COMPOSITE_"):
            return state_tuple[0]
        
        # Handle padding
        if state_tuple == self.pad_symbol:
            return "PAD"
        # Handle the case where a state is empty, representing preparation
        if not state_tuple or state_tuple == ('grasper-null_verb-null_target',):
            return "ATOMIC__preparation"
        # Handle concurrent/atomic states
        prefix = "CONCURRENT__" if len(state_tuple) > 1 else "ATOMIC__"
        body = '_AND_'.join(state_tuple).replace(',', '-').replace(' ', '_')
        return prefix + body

    def _parse_sequences_from_file(self, corpus_filepath):
        """Parses the raw sequence file into a list of lists of states."""
        if not os.path.exists(corpus_filepath):
            raise FileNotFoundError(f"Corpus file not found: {corpus_filepath}")
        
        all_sequences = []
        for line in open(corpus_filepath, 'r'):
            
            line = line.strip('->')
            
            if not line: continue
            # print(line)
            matches = re.findall(r'\[([^\]]+)\]\(([\d.]+)s\)', line)
            # matches = re.findall(r'\[([^\]]+)\]\((\d+)s\)', line)
            video_sequence = []
            # print(matches)
            for action_str, duration_str in matches:
                # actions = tuple(sorted(action_str.split(' AND ')))
                duration = float(duration_str)
                actions = ('preparation',) if action_str == 'preparation' else tuple(sorted(action_str.split(' AND ')))
                # video_sequence.append({'actions': actions, 'duration': int(duration_str)})
                video_sequence.append({'actions': actions, 'duration': duration})
            all_sequences.append(video_sequence)
        return all_sequences

    def _mine_frequent_ngrams(self, sequences, n, min_support):
        """
        Finds frequent contiguous sub-sequences of actions (n-grams).
        Note: For very large corpora, a more optimized algorithm like a suffix tree
        or a sliding window with collections.deque could improve performance.
        """
        counts = Counter()
        for seq in sequences:
            action_only_seq = [block['actions'] for block in seq]
            for i in range(len(action_only_seq) - n + 1):
                ngram = tuple(action_only_seq[i:i+n])
                counts[ngram] += 1
        return {ng: c for ng, c in counts.items() if c >= min_support}

    def _replace_with_composites(self, sequences, composite_map):
        """
        Replaces frequent n-grams with composite tokens using a greedy, left-to-right
        maximal match strategy. The longest possible composite is replaced first at any position.
        """
        new_sequences = []
        # Sort composites by length (desc) to replace longest ones first at any given position
        sorted_composites = sorted(composite_map.items(), key=lambda x: len(x[0]), reverse=True)

        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                matched = False
                for group, name in sorted_composites:
                    n = len(group)
                    if i + n > len(seq): continue
                    
                    action_group = tuple(s['actions'] for s in seq[i:i+n])
                    if action_group == group:
                        composite_duration = sum(s['duration'] for s in seq[i:i+n])
                        new_seq.append({'actions': (name,), 'duration': composite_duration})
                        i += n
                        matched = True
                        break
                if not matched:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        return new_sequences

    def _mine_chiral_pairs(self, sequences, min_support):
        """Specifically finds frequent 2-grams where the verbs are chiral opposites."""
        counts = Counter()
        for seq in sequences:
            action_only_seq = [block['actions'] for block in seq]
            for i in range(len(action_only_seq) - 1):
                from_state = action_only_seq[i]
                to_state = action_only_seq[i+1]
                
                from_verb = self._get_verb_from_state(from_state)
                to_verb = self._get_verb_from_state(to_state)
                
                if from_verb and to_verb:
                    if self.chirality_map_full.get(from_verb) == to_verb:
                        counts[(from_state, to_state)] += 1
                        
        return {ng: c for ng, c in counts.items() } #if c >= min_support


    def analyze_chirality(self):
        """Analyzes learned transitions and durations for chiral properties."""
        chiral_transitions = defaultdict(Counter)
        total_from_counts = Counter()
        for history, next_states in self.transition_counts.items():
            if history == self.start_symbol: continue
            from_state = history[-1]
            from_verb = self._get_verb_from_state(from_state)
            if from_verb and from_verb in self.chirality_map_full:
                total_from_counts[from_verb] += sum(next_states.values())
                for to_state, count in next_states.items():
                    to_verb = self._get_verb_from_state(to_state)
                    # print("from_verb,to_verb , from_state, to_state", from_verb,to_verb , from_state, to_state )
                    if to_verb and to_verb == self.chirality_map_full[from_verb]:
                        chiral_transitions[from_verb][to_verb] += count
        
        chiral_report = {}
        for from_verb, to_verbs in chiral_transitions.items():
            to_verb = self.chirality_map_full[from_verb]
            prob = to_verbs[to_verb] / total_from_counts[from_verb] if total_from_counts[from_verb] > 0 else 0
            chiral_report[from_verb] = {'opposite': to_verb, 'prob': prob}
        return chiral_report  


    def learn(self, corpus_filepath):
        """Main function to learn the hierarchical grammar."""
        print("--- Learning Hierarchical Grammar ---")
        sequences = self._parse_sequences_from_file(corpus_filepath)
        print("sequences",sequences)
        print("Step 1: Mining for frequent n-grams to create composite actions...")
        current_sequences = sequences
        for n in range(2, MAX_NGRAM_SIZE + 1):
            print(f"  - Mining for {n}-grams...")
            frequent_ngrams = self._mine_frequent_ngrams(current_sequences, n, MIN_NGRAM_FREQUENCY)
            if not frequent_ngrams:
                print(f"  - No frequent {n}-grams found. Stopping hierarchy building.")
                break
            
            new_composites = {}
            for ngram in frequent_ngrams:
                # Use a unique ID for the name and store the alias
                composite_id = f"COMPOSITE_{len(self.composite_definitions) + 1}"
                readable_alias = "_then_".join([self._get_state_name(state) for state in ngram])
                self.composite_definitions[composite_id] = (ngram, readable_alias)
                new_composites[ngram] = composite_id
            
            print(f"  - Found {len(frequent_ngrams)} frequent {n}-grams. Replacing them in sequences.")
            current_sequences = self._replace_with_composites(current_sequences, new_composites)

        print(f"\nStep 2: Learning {MARKOV_ORDER}-order transition probabilities and duration models...")

        print("\nStep 2: Mining for frequent chiral pair transitions...")
        print("sequences",sequences)
        chiral_ngrams = self._mine_chiral_pairs(sequences, min_support=1)  # override frequency
        self.chiral_ngrams = chiral_ngrams
        print("chiral_ngrams",chiral_ngrams)
        
        if chiral_ngrams:
            for (from_state, to_state), count in chiral_ngrams.items():
              # add smoothing: guarantee that chiral transition exists
              self.transition_counts[(from_state,)][to_state] += max(count, 1)
            new_chiral_composites = {}
            for ngram in chiral_ngrams:
                composite_id = f"CHIRAL_PAIR_{len(self.composite_definitions) + 1}"
                alias = "_then_".join([self._get_state_name(state) for state in ngram])
                self.composite_definitions[composite_id] = (ngram, alias)
                new_chiral_composites[ngram] = composite_id
            
            print(f"  - Found {len(chiral_ngrams)} frequent chiral pairs. Integrating into sequences.")
            current_sequences = self._replace_with_composites(current_sequences, new_chiral_composites)
        else:
            print("  - No frequent chiral pairs found.")

        for video_sequence in current_sequences:
            actions_only = [block['actions'] for block in video_sequence]
            padded_sequence = [self.pad_symbol] * MARKOV_ORDER + actions_only
            
            for i in range(len(padded_sequence) - MARKOV_ORDER):
                history = tuple(padded_sequence[i : i + MARKOV_ORDER])
                next_state = padded_sequence[i + MARKOV_ORDER]
                
                is_start_transition = all(s == self.pad_symbol for s in history[1:])
                if is_start_transition:
                     self.transition_counts[self.start_symbol][next_state] += 1
                else:
                    self.transition_counts[history][next_state] += 1
            
            for block in video_sequence:
                self.duration_data[block['actions']].append(block['duration'])
                self.state_occurrence_count[block['actions']] += 1
        print("Grammar learning complete.")

    def save_grammar(self, output_filepath):
        """Saves the learned hierarchical grammar to a file with validation."""
        print(f"--- Saving Hierarchical Grammar to '{output_filepath}' ---")
        with open(output_filepath, 'w') as f:
            f.write("### Hierarchical Probabilistic Surgical Grammar with Durations ###\n\n")
            f.write(f"# Markov Order: {MARKOV_ORDER}\n\n")
            
            f.write("="*25 + " COMPOSITE RULES " + "="*25 + "\n")
            for name, (group, alias) in self.composite_definitions.items():
                body = " ".join([self._get_state_name(state) for state in group])
                f.write(f"# Alias: {alias}\n{name} -> {body} [1.0]\n")
            f.write("\n")

            f.write("="*25 + " GRAMMAR RULES " + "="*25 + "\n")
            
            all_histories = {self.start_symbol}.union(set(self.transition_counts.keys()))
            
            for history in sorted(list(all_histories),  key=str):
                transitions = self.transition_counts.get(history, {})
                if not transitions: continue

                if history == self.start_symbol:
                    from_state_name = self.start_symbol
                    f.write(f"# Transitions from START symbol (S)\n")
                else:
                    from_state_name = " | ".join([self._get_state_name(s) for s in history])
                    f.write(f"# Transitions from state sequence: ({from_state_name})\n")
                
                total_transitions = sum(transitions.values())
                total_prob = sum(count / total_transitions for count in transitions.values())
                assert abs(total_prob - 1.0) < 1e-6, f"Probabilities for {from_state_name} do not sum to 1, but to {total_prob}"

                for to_state, count in transitions.items():
                    prob = count / total_transitions
                    to_state_name = self._get_state_name(to_state)
                    f.write(f"{from_state_name} -> {to_state_name} [{prob:.5f}]\n")
                f.write("\n")
            
            f.write("="*25 + " TERMINAL RULES " + "="*25 + "\n")
            all_atomic_states = {s for s in self.state_occurrence_count if not s[0].startswith("COMPOSITE_")}
            for state in sorted(list(all_atomic_states)):
                 state_name = self._get_state_name(state)
                 f.write(f"{state_name} -> '{state_name}' [1.0]\n")
            f.write("\n")
            
            f.write("="*25 + " DURATION MODEL " + "="*25 + "\n")
            f.write("# Format: StateName | Mean (μ) | Std Dev (σ) | Median | MAD | Occurrences\n\n")
            
            sorted_durations = sorted(self.duration_data.items(), key=lambda item: len(item[1]), reverse=True)
            for state, durations in sorted_durations:
                state_name = self._get_state_name(state)
                durations_np = np.array(durations)
                mean_dur = np.mean(durations_np)
                # Use sample standard deviation (ddof=1) for robustness with small samples
                std_dev = np.std(durations_np, ddof=1) if len(durations) > 1 else 0.0
                median_dur = np.median(durations_np)
                mad = np.median(np.abs(durations_np - median_dur))
                f.write(f"{state_name} | {mean_dur:.2f} | {std_dev:.2f} | {median_dur:.2f} | {mad:.2f} | {len(durations)}\n")

            # --- NEW CHIRALITY SECTION ---
            f.write("\n" + "="*25 + " CHIRALITY ANALYSIS " + "="*25 + "\n")
            f.write("# Analysis of temporally opposite action pairs.\n\n")
            chiral_analysis = self.analyze_chirality()
            f.write(f"{'Action (Verb)':<25} | {'Opposite Action':<25} | {'Transition Prob.':<20} | {'Duration Stats (μ ± σ)'}\n")
            f.write("-" * 120 + "\n")
            
            for from_verb, data in sorted(chiral_analysis.items()):
                print(from_verb, data)
                to_verb = data['opposite']
                prob_str = f"{data['prob']:.3f}"
                
                from_states = [s for s in self.duration_data if self._get_verb_from_state(s) == from_verb]
                to_states = [s for s in self.duration_data if self._get_verb_from_state(s) == to_verb]
                
                from_durations = [d for s in from_states for d in self.duration_data[s]]
                to_durations = [d for s in to_states for d in self.duration_data[s]]
                
                from_stats = f"{np.mean(from_durations):.1f} ± {np.std(from_durations, ddof=1):.1f}s" if from_durations else "N/A"
                to_stats = f"{np.mean(to_durations):.1f} ± {np.std(to_durations, ddof=1):.1f}s" if to_durations else "N/A"
                
                f.write(f"{from_verb:<25} | {to_verb:<25} | {prob_str:<20} | {from_stats} vs {to_stats}\n")
                expected_ratio = np.mean(to_durations) / max(np.mean(from_durations), 1e-6)
                f.write(f"# Chirality temporal ratio {from_verb}->{to_verb}: {expected_ratio:.2f}\n")

                for (from_state, to_state), count in self.chiral_ngrams.items():
                    from_state_name = self._get_state_name(from_state)
                    to_state_name = self._get_state_name(to_state)
                    f.write(f"{from_state_name} -> {to_state_name} [CHIRAL_PRIOR]\n")
        print("Grammar and chirality analysis saved successfully.")



def main():
    learner = HierarchicalGrammarLearner()
    learner.learn(SEQUENCE_CORPUS_FILE)
    learner.save_grammar(OUTPUT_GRAMMAR_FILE)
    print(f"\nProcess complete. Hierarchical grammar is in '{OUTPUT_GRAMMAR_FILE}'.")

if __name__ == "__main__":
    main()