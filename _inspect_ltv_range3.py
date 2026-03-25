import pandas as pd

df_hf = pd.read_excel('C:/Users/rochant/WorkBuddy/Claw/DataFusion/LTV_High-Fidelity.xlsx')

# 筛选训练集条件: 1.2 < Mach < 3.5 且 7.0 < Alpha < 15.0
mask_train = (df_hf['Mach'] > 1.2) & (df_hf['Mach'] < 3.5) & \
             (df_hf['Alpha'] > 7.0) & (df_hf['Alpha'] < 15.0)
df_hf_train = df_hf[mask_train].copy()
df_hf_test = df_hf[~mask_train].copy()

print(f'HF 总条数: {len(df_hf)}')
print(f'HF 训练集条数 (1.2<Mach<3.5 且 7.0<Alpha<15.0): {len(df_hf_train)}')
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
