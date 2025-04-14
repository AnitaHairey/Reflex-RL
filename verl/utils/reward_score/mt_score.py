import re
from typing import Dict, Tuple, Optional
import sacrebleu

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

def extract_solution(solution_str: str) -> Tuple[str, str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
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
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<translate>(.*?)</translate>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<translate>', 1),
        'answer_end': ('</translate>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
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
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing Training Sample ".center(80, '='))
    

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    # Validate response structure
    if check_think:
        format_correct = validate_response_structure(processed_str)
        format_score = format_reward if format_correct else -abs(format_reward)
    else:
        format_correct = answer_text != None
        format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")


    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:

        bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)

        if reward_type == 'discrete':
            if reward_metric == 'BLEU':
                if bleu_score > bleu_threshold:
                    answer_score = 2
                else:
                    answer_score = -1.5

            elif reward_metric == 'Model':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                if metric_score > comet_threshold:
                    answer_score = 2
                else:
                    answer_score = -1.5

            elif reward_metric == 'Merge':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                if bleu_score > bleu_threshold and metric_score > comet_threshold:
                    answer_score = 2
                else:
                    answer_score = -1.5

        elif reward_type == 'continuous':
            if reward_metric == 'BLEU':
                answer_score = float(bleu_score) / float(scale_factor)

            elif reward_metric == 'Model':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                answer_score = float(metric_score) / float(scale_factor)


            elif reward_metric == 'Merge':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                answer_score = float(bleu_score) / float(scale_factor) + float(metric_score) / float(scale_factor)

        else:
            raise ValueError("Invalid reward_type, please use discrete or continuous")

        print(f"\n[Content Validation]")
        print(f"Reference: {ground_truth}")
        print(f"Hypothesis: {answer_text}")
        print(f"BLEU Score: {bleu_score}")
        print(f"Metric Model Score: {metric_score}" if metric_score is not None else "") 
        
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Reward Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score



def compute_score_val_bleu(solution_str: str, 
                 ground_truth: str,
                 lg_pair:str, 
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing Test Sample ".center(80, '='))
    
    solution_text = ground_truth
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    if answer_text:
        pred_status = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {answer_text}")
    else:
        pred_status = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {processed_str}")
        

    total_score = pred_status
    print("\n" + "-"*80)
    print(f"BLEU Score: {total_score}")


    return total_score