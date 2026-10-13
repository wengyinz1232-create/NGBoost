"""Create user-facing Chinese interpretation and plots from verified evidence only."""
from pathlib import Path
import json,csv,statistics as st,os,sys
sys.stdout.reconfigure(encoding='utf-8')
root=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(out/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=json.loads((root/'reports/comparison.json').read_text())
assert summary['status']=='PASS' and summary['folds']==10
docs=[json.loads(p.read_text()) for p in sorted((root/'comparison').glob('fold_*/result.json'))]
assert len(docs)==10
conditions=['clean','white_20','white_10','pink_10']
labels=['Clean','White 20 dB','White 10 dB','Pink 10 dB']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for name,color in [('ngboost','#bd6427'),('lightgbm','#187b99')]:
 a=[[d['models'][name]['metrics'][c]['macro_f1'] for d in docs] for c in conditions]
 axes[0].errorbar(range(4),[st.mean(x) for x in a],yerr=[st.stdev(x) for x in a],marker='o',capsize=3,label=name,color=color)
axes[0].set_xticks(range(4),labels,rotation=15);axes[0].set_ylim(0,1.05)
axes[0].set_ylabel('Macro-F1 (mean +/- fold SD)');axes[0].legend();axes[0].grid(alpha=.2)
diff=next(p for p in summary['paired'] if p['condition']=='pink_10' and p['metric']=='macro_f1')
axes[1].bar(range(1,11),diff['fold_differences'],color=['#bd6427' if d>0 else '#187b99' for d in diff['fold_differences']])
axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(range(1,11));axes[1].set_xlabel('Paired outer fold')
axes[1].set_ylabel('Pink F1: NGBoost minus LightGBM');axes[1].grid(axis='y',alpha=.2)
fig.suptitle('MFCC+delta / white-noise augmentation / ten unseen-speaker folds')
fig.savefig(out/'ngboost_lightgbm_comparison.png',dpi=200);plt.close(fig)
lines=['# NGBoost 与 LightGBM：实际训练结果','',
'范围：MFCC+delta，干净＋10 dB 白噪声增强；外层十折、内层三折，测试说话人未参与训练。每对模型在同一 GitHub runner 顺序运行。',
'','所有均值与 SD 均由十个外折计算；SD 不是独立置信区间。差值为 NGBoost − LightGBM。',
'','| 条件 | NGBoost macro-F1 | LightGBM macro-F1 | 配对差值 | NGBoost 胜出 |','|---|---:|---:|---:|---:|']
for c,label in zip(conditions,labels):
 def ms(name,metric):
  a=[d['models'][name]['metrics'][c][metric] for d in docs]
  return f'{st.mean(a):.4f} ± {st.stdev(a):.4f}'
 p=next(p for p in summary['paired'] if p['condition']==c and p['metric']=='macro_f1')
 lines.append(f"| {label} | {ms('ngboost','macro_f1')} | {ms('lightgbm','macro_f1')} | {p['mean']:+.4f} ± {p['fold_sd']:.4f} | {p['ngboost_wins']}/10 |")
lines+=['','## 概率质量与成本','', '| 指标 | NGBoost | LightGBM |','|---|---:|---:|']
c='pink_10'
for metric in ['accuracy','macro_precision','log_loss']:
 lines.append(f"| Pink-10 {metric} | {ms('ngboost',metric)} | {ms('lightgbm',metric)} |")
for metric,label in [('final_fit_seconds_mean','最终拟合秒/折'),('workflow_seconds_mean','完整流程秒/折'),('model_bytes_mean','保存模型字节/折')]:
 lines.append(f"| {label} | {summary['costs']['ngboost'][metric]:.2f} | {summary['costs']['lightgbm'][metric]:.2f} |")
ratio=summary['costs']['ngboost']['final_fit_seconds_mean']/summary['costs']['lightgbm']['final_fit_seconds_mean']
lines+=['',f'NGBoost / LightGBM 的平均最终拟合时间比为 {ratio:.2f}。完整流程还包括内层搜索、验证预测、保存与重载核验，两种方法的实际拟合调用次数不同；不能把流程时间当作一次拟合时间。',
'','## 稳定性和支持范围','']
if diff['mean']<0:
 lines.append(f"NGBoost 的平均粉红噪声 macro-F1 低于 LightGBM；十折中 NGBoost 胜出 {diff['ngboost_wins']} 折，差值范围 [{diff['min']:+.4f}, {diff['max']:+.4f}]。这是当前配置的观测结果，不能推广为 NGBoost 算法普遍更差；小均值差或折间胜负混合时，尤其不能仅凭均值宣称可靠优势。")
elif diff['mean']>0:
 lines.append(f"NGBoost 的平均粉红噪声 macro-F1 高于 LightGBM，胜出 {diff['ngboost_wins']}/10 折；因此不能用这次结果支持 NGBoost 预测更差的预设结论。")
else:lines.append('两者平均粉红噪声 macro-F1 相同，未建立均值优势。')
loss=next(p for p in summary['paired'] if p['condition']=='pink_10' and p['metric']=='log_loss')
lines.append(f"粉红噪声 log loss 的配对差值为 {loss['mean']:+.4f} ± {loss['fold_sd']:.4f}（越低越好），NGBoost 在 {loss['ngboost_wins']}/10 折胜出。log loss 不能单独证明校准。")
ng_sd=st.stdev(d['models']['ngboost']['metrics']['pink_10']['macro_f1'] for d in docs)
lg_sd=st.stdev(d['models']['lightgbm']['metrics']['pink_10']['macro_f1'] for d in docs)
lines.append(f'粉红噪声 F1 的折间 SD 为 NGBoost {ng_sd:.4f}、LightGBM {lg_sd:.4f}。本次没有失败折，两种模型的训练警告记录均为零；不能将较慢或较低均值等同于训练发散或普遍更不稳定。')
lines+=['','## 参数审查和局限','']
for name in ['ngboost','lightgbm']:
 a=summary['parameter_audit'][name]
 lines.append(f"- {name}：网格 {a['grid']}，选择计数 {a['counts']}；中间值唯一众数：{a['middle_unique_mode']}；记录警告数：{a['warnings']}。")
 if name=='ngboost' and a['counts'][-1]>0:
  lines.append('  NGBoost 存在选中最大轮数的折，说明此预算内可能仍受轮数限制；不能声称已经找到其最佳配置。若调整范围，需要在内层重新选择并对再次探索的局限作说明，最好采用新的外部测试。')
  if not a['middle_unique_mode']:
   lines.append('  该 NGBoost 配置未通过课程“中间参数最常选”的检查，应作为披露局限的补充对比，不能标为全面合规或据此替换已核验的正式三模型。')
lines+=['','这是一个外层划分、固定噪声样本、单种特征与增强方式的比较。未测峰值内存、实际设备延迟或电量，也未做真实背景声和新麦克风测试。自然梯度、线搜索、树数量和实现差异可能解释成本，但本实验没有拆分这些机制，不能将耗时差全部归因于某一项。',
'','原三模型主实验保持不变；这里训练二十个外折模型，没有为 NGBoost 另外训练全数据部署模型。',
'','数据文件：summary.csv、paired_differences.csv、comparison.json、逐折 result.json 及 Actions 工件中的预测和模型。']
(out/'NGBOOST_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
