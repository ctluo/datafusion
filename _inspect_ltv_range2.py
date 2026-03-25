import pandas as pd

df_hf = pd.read_excel('C:/Users/rochant/WorkBuddy/Claw/DataFusion/LTV_High-Fidelity.xlsx')

# 筛选训练集条件: 1.6 < Mach < 2.8 且 3.0 < Alpha < 17.0
mask_train = (df_hf['Mach'] > 1.6) & (df_hf['Mach'] < 2.8) & \
             (df_hf['Alpha'] > 3.0) & (df_hf['Alpha'] < 17.0)
df_hf_train = df_hf[mask_train].copy()
df_hf_test = df_hf[~mask_train].copy()

print(f'HF 总条数: {len(df_hf)}')
print(f'HF 训练集条数 (1.6<Mach<2.8 且 3.0<Alpha<17.0): {len(df_hf_train)}')
print(f'HF 测试集条数 (其余): {len(df_hf_test)}')
print()
print('训练集 Mach/Alpha 范围:')
print(f'  Mach: [{df_hf_train["Mach"].min()}, {df_hf_train["Mach"].max()}]')
print(f'  Alpha: [{df_hf_train["Alpha"].min()}, {df_hf_train["Alpha"].max()}]')
print()
print('测试集 Mach/Alpha 范围:')
print(f'  Mach: [{df_hf_test["Mach"].min()}, {df_hf_test["Mach"].max()}]')
print(f'  Alpha: [{df_hf_test["Alpha"].min()}, {df_hf_test["Alpha"].max()}]')
print()
print('训练集:')
print(df_hf_train)
print()
print('测试集:')
print(df_hf_test)
