# batch推理，使用平均得分为ground truth

from mathruler.grader import extract_boxed_content


# scores = self.compute_score(
#     data_sources=data_sources,
#     solution_strs=responses_str,
#     ground_truths=ground_truths,
#     extra_infos=extras,
#     **self.reward_kwargs,
# )


def compute_score(data_sources: list[str], solution_strs: list[str], ground_truths: list, extra_infos: list, **kwargs) -> list[float]:
    index_groups = {}
    for sol_str, info in zip(solution_strs, extra_infos):
        index = info['index']
        score = extract_boxed_content(sol_str)
        if index not in index_groups:
            index_groups[index] = []
        index_groups[index].append(score)
    
    group_stats = {}
    for idx, scores in index_groups.items():
        valid_scores = []
        for score in scores:
            try:
                valid_scores.append(float(score))
            except ValueError:
                # 处理无法转换为浮点数的情况
                pass
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        group_stats[idx] = mean_score
    
    grouped_scores = []
    for sol_str, info in zip(solution_strs, extra_infos):
        index = info['index']
        mean_score = group_stats[index]
        try:
            grouped_scores.append(1 - abs(float(extract_boxed_content(sol_str)) - mean_score) / 10)
        except ValueError:
            # 处理无法转换为浮点数的情况
            grouped_scores.append(0.0)
    
    
    return grouped_scores
