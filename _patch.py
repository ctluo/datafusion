src = open(r'C:\Users\rochant\WorkBuddy\Claw\DataFusion\benchmark_fusion_50pct.py', encoding='utf-8').read()
dst = (src
    .replace('50% 训练 / 50% 测试', '80% 训练 / 20% 测试')
    .replace('HF 50% Train / 50% Test', 'HF 80% Train / 20% Test')
    .replace('test_size=0.50', 'test_size=0.20')
    .replace("条  (50%)\"))\nprint(f\"  高置信度测试集   : {X_hf_test.shape[0]} 条  (50%)",
           "条  (80%)\"))\nprint(f\"  高置信度测试集   : {X_hf_test.shape[0]} 条  (20%)")
    .replace('测试集: HF 50%', '测试集: HF 20%')
    .replace('HF Train: 50% | HF Test: 50%', 'HF Train: 80% | HF Test: 20%')
    .replace('fusion_benchmark_50pct', 'fusion_benchmark_80pct')
)
open(r'C:\Users\rochant\WorkBuddy\Claw\DataFusion\benchmark_fusion_80pct.py', 'w', encoding='utf-8').write(dst)
print('done')
