import re
from typing import Dict, Tuple, Optional
import sacrebleu
from collections import Counter

def compute_ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Compute n-gram overlap F1 score between two texts.
    
    Args:
        text1: First text
        text2: Second text  
        n: N-gram size (default: 2)
        
    Returns:
        F1 score of n-gram overlap
    """
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return Counter(ngrams)
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    # Calculate intersection
    intersection = sum((ngrams1 & ngrams2).values())
    
    # Calculate precision and recall
    precision = intersection / sum(ngrams1.values()) if sum(ngrams1.values()) > 0 else 0.0
    recall = intersection / sum(ngrams2.values()) if sum(ngrams2.values()) > 0 else 0.0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def compute_bleu(lg_pair, ref, pred):
    # 新增类型检查
    pred = pred if isinstance(pred, str) else ""
    
    src_lang = lg_pair.split("-")[0]
    tgt_lang = lg_pair.split("-")[1]
    
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"
    
    refs = [[ref]]
    sys = [pred]

    bleu_str = str(sacrebleu.corpus_bleu(sys, refs, tokenize=tokenize))  # 注意bleu tokenize
    bleu_score = re.search(r'BLEU = (\d+\.\d+)', bleu_str).group(1)

    print(f"[BLEU Score] {bleu_score}")
    return float(bleu_score)

def extract_solution(solution_str: str) -> Tuple[str, str, str, str]:
    """Extracts the draft, reflection, and final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (draft_answer, reflection_text, final_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str: # base
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str: # qwen and tower
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str: # llama3
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, None, None, solution_str

    # Extract draft translation
    draft_pattern = r'<draft>(.*?)</draft>'
    draft_matches = list(re.finditer(draft_pattern, processed_str, re.DOTALL))
    draft_answer = draft_matches[-1].group(1).strip() if draft_matches else None

    # Extract reflection
    reflection_pattern = r'<reflection>(.*?)</reflection>'
    reflection_matches = list(re.finditer(reflection_pattern, processed_str, re.DOTALL))
    reflection_text = reflection_matches[-1].group(1).strip() if reflection_matches else None

    # Extract final translation
    final_pattern = r'<final>(.*?)</final>'
    final_matches = list(re.finditer(final_pattern, processed_str, re.DOTALL))
    final_answer = final_matches[-1].group(1).strip() if final_matches else None

    if not final_answer:
        print("[Error] No valid answer tags found")
        return None, None, None, processed_str
        
    return draft_answer, reflection_text, final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of Reflex-RL response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags for reflex format
    tags = {
        'draft_start': ('<draft>', 1),
        'draft_end': ('</draft>', 1),
        'reflection_start': ('<reflection>', 1),
        'reflection_end': ('</reflection>', 1),
        'final_start': ('<final>', 1),
        'final_end': ('</final>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order: draft -> reflection -> final
    if validation_passed and all(pos >= 0 for pos in positions.values()):
        if (positions['draft_start'] > positions['draft_end'] or
            positions['draft_end'] > positions['reflection_start'] or
            positions['reflection_start'] > positions['reflection_end'] or
            positions['reflection_end'] > positions['final_start'] or
            positions['final_start'] > positions['final_end']):
            print("  [Error] Incorrect tag order: Expected <draft>...</draft><reflection>...</reflection><final>...</final>")
            validation_passed = False
        else:
            print("  Tag sequence validation passed")

    return validation_passed




def compute_score(reward_metric: str,
                 reward_type: str,
                 metric_score: None,
                 lg_pair: None,
                 bleu_threshold: float,
                 comet_threshold: float,
                 solution_str: str, 
                 ground_truth: str,
                 scale_factor: float = 100.0,
                 check_think: bool = True,
                 format_reward: int = 1,
                 w_final: float = 1.0,
                 w_improve: float = 1.0, 
                 w_critique: float = 1.0) -> float:
    """Computes comprehensive score for model response using Reflex-RL framework.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        w_final: Weight for final quality reward
        w_improve: Weight for improvement gain reward
        w_critique: Weight for critique grounding reward
        
    Returns:
        Total weighted score (r_final + r_improve + r_critique)
    """
    print("\n" + "="*80)
    print(" Processing Training Sample (Reflex-RL) ".center(80, '='))
    
    # Extract draft, reflection, and final answer for Reflex-RL
    draft_answer, reflection_text, final_answer, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    # Validate response structure  
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Initialize reward components
    r_final = 0.0
    r_improve = 0.0
    r_critique = 0.0

    if format_correct and final_answer:
        # 1. Final Quality Reward (r_final)
        if reward_type == 'discrete':
            if reward_metric == 'BLEU':
                final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
                r_final = 2 if final_bleu > bleu_threshold else -1.5
            elif reward_metric == 'Model':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                r_final = 2 if metric_score > comet_threshold else -1.5
            elif reward_metric == 'Merge':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
                r_final = 2 if (final_bleu > bleu_threshold and metric_score > comet_threshold) else -1.5

        elif reward_type == 'continuous':
            if reward_metric == 'BLEU':
                final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
                r_final = float(final_bleu) / float(scale_factor)
            elif reward_metric == 'Model':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                r_final = float(metric_score) / float(scale_factor)
            elif reward_metric == 'Merge':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
                r_final = (float(final_bleu) + float(metric_score)) / float(scale_factor)

        # 2. Improvement Gain Reward (r_improve) - Core innovation of Reflex-RL
        if draft_answer and final_answer:
            draft_bleu = compute_bleu(lg_pair, ground_truth, draft_answer)
            final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
            
            if reward_type == 'continuous':
                if reward_metric in ['BLEU', 'Merge']:
                    r_improve = (final_bleu - draft_bleu) / float(scale_factor)
                elif reward_metric == 'Model':
                    # For model-based metrics, we approximate improvement using BLEU
                    # In practice, you might want to compute metric scores for both draft and final
                    r_improve = (final_bleu - draft_bleu) / float(scale_factor)
            elif reward_type == 'discrete':
                improvement = final_bleu - draft_bleu
                r_improve = 1.0 if improvement > 1.0 else -0.5 if improvement < -1.0 else 0.0

            print(f"[Improvement Analysis]")
            print(f"Draft BLEU: {draft_bleu:.2f}")
            print(f"Final BLEU: {final_bleu:.2f}")
            print(f"Improvement: {final_bleu - draft_bleu:.2f}")

        # 3. Critique Grounding Reward (r_critique)
        if reflection_text and draft_answer:
            # Compute n-gram overlap between reflection and context (source + draft)
            source_text = solution_str.split("User:")[1].split("Assistant:")[0].strip() if "User:" in solution_str else ""
            overlap_with_source = compute_ngram_overlap(reflection_text, source_text, n=2)
            overlap_with_draft = compute_ngram_overlap(reflection_text, draft_answer, n=2)
            r_critique = (overlap_with_source + overlap_with_draft) / 2.0
            
            print(f"[Critique Analysis]")
            print(f"Reflection text: {reflection_text[:100]}...")
            print(f"Overlap with source: {overlap_with_source:.3f}")
            print(f"Overlap with draft: {overlap_with_draft:.3f}")

        print(f"\n[Content Validation]")
        print(f"Reference: {ground_truth}")
        print(f"Draft: {draft_answer}")
        print(f"Final: {final_answer}")
        if metric_score is not None:
            print(f"Metric Model Score: {metric_score}")
    else:
        r_final = -2
        r_improve = -1  
        r_critique = -1
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    # Compute total weighted reward
    total_score = w_final * r_final + w_improve * r_improve + w_critique * r_critique
    
    print("\n" + "-"*80)
    print(f" Reflex-RL Reward Breakdown ".center(80, '-'))
    print(f"  Final Quality (r_final): {r_final:.3f} (weight: {w_final})")
    print(f"  Improvement Gain (r_improve): {r_improve:.3f} (weight: {w_improve})")
    print(f"  Critique Grounding (r_critique): {r_critique:.3f} (weight: {w_critique})")
    print(f"  Total Weighted Score: {total_score:.3f}")
    print("="*80 + "\n")

    return total_score



def compute_score_val_bleu(solution_str: str, 
                 ground_truth: str,
                 lg_pair:str,
                 format_reward: int = 1,
                 answer_reward: float = 1.0) -> float:
    """Computes BLEU score for model response during validation using Reflex-RL format.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        BLEU score of the final translation
    """
    print("\n" + "="*80)
    print(" Processing Test Sample ".center(80, '='))
    
    solution_text = ground_truth
    
    # Extract draft, reflection, and final answer from Reflex-RL format
    draft_answer, reflection_text, final_answer, processed_str = extract_solution(solution_str)
    answer_text = final_answer if final_answer else draft_answer
    
    print(f"\n[Prompt + Response]\n{solution_str}")

    if answer_text:
        pred_status = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {answer_text}")
        if draft_answer and final_answer and draft_answer != final_answer:
            draft_bleu = compute_bleu(lg_pair, ground_truth, draft_answer)
            print(f"Draft: {draft_answer}")
            print(f"Draft BLEU: {draft_bleu:.2f}")
            print(f"Improvement: {pred_status - draft_bleu:.2f}")
    else:
        pred_status = compute_bleu(lg_pair, ground_truth, "")
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {processed_str}")
        

    total_score = pred_status
    print("\n" + "-"*80)
    print(f"BLEU Score: {total_score}")

    return total_score