import ast
with open(r'C:\Users\rochant\WorkBuddy\Claw\DataFusion\benchmark_fusion_80pct.py', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print('syntax OK')
except SyntaxError as e:
    print('SyntaxError:', e)

# 快速检查关键字段
for kw in ['test_size=0.20', '80%', '20%', 'benchmark_80pct']:
    print(kw, ':', kw in src)
