"""Build the Chinese coursework PDF from recorded experiment artifacts.

Run with Codex bundled Python (reportlab, pypdf, pandas). Source numbers are
loaded from CSV/JSON so the report can be regenerated after another iteration.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output';FIG=OUT/'figures'
DEST=OUT/'pdf';DEST.mkdir(exist_ok=True)
FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
SONG='/System/Library/Fonts/Supplemental/Songti.ttc'
if Path(FONT).exists():pdfmetrics.registerFont(TTFont('CN',FONT))
else:pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'));FONT='STSong-Light'
if Path(SONG).exists():pdfmetrics.registerFont(TTFont('TitleCN',SONG))
else:SONG='CN'
BODY='CN' if FONT!='STSong-Light' else 'STSong-Light'
TITLE='TitleCN' if SONG!='CN' else BODY
NAVY=colors.HexColor('#263c50');TEAL=colors.HexColor('#328579');GRAY=colors.HexColor('#edf1f4')
ST={
 'body':ParagraphStyle('body',fontName=BODY,fontSize=10.1,leading=16.4,spaceAfter=7,wordWrap='CJK',textColor=colors.HexColor('#222222')),
 'h1':ParagraphStyle('h1',fontName=TITLE,fontSize=16.5,leading=23,spaceAfter=12,textColor=NAVY),
 'h2':ParagraphStyle('h2',fontName=TITLE,fontSize=12,leading=18,spaceBefore=4,spaceAfter=7,textColor=NAVY),
 'small':ParagraphStyle('small',fontName=BODY,fontSize=8.4,leading=12,spaceAfter=6,wordWrap='CJK',textColor=colors.HexColor('#555555')),
 'table':ParagraphStyle('table',fontName=BODY,fontSize=8.6,leading=12.2,wordWrap='CJK'),
 'table_small':ParagraphStyle('table_small',fontName=BODY,fontSize=7.6,leading=10.4,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName=BODY,fontSize=8.6,leading=12.5,alignment=TA_CENTER,spaceAfter=9,textColor=colors.HexColor('#555555')),
 'title':ParagraphStyle('title',fontName=TITLE,fontSize=25,leading=38,alignment=TA_CENTER,textColor=NAVY),
 'cover':ParagraphStyle('cover',fontName=TITLE,fontSize=16,leading=28,alignment=TA_CENTER),
 'formula':ParagraphStyle('formula',fontName='Helvetica',fontSize=10,leading=16,spaceAfter=9,alignment=TA_CENTER),
}
story=[]
def P(text,style='body'):
    return Paragraph(text,ST[style])
def para(text,style='body'):story.append(P(text,style))
def section(text):para(text,'h1')
def sub(text):para(text,'h2')
def page():story.append(PageBreak())
def table(headers,rows,widths=None,small=False):
    sty='table_small' if small else 'table'
    data=[[P(escape(str(x)),sty) for x in headers]]+[[P(escape(str(x)),sty) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='CENTER')
    t.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0),GRAY),('TEXTCOLOR',(0,0),(-1,0),NAVY),
      ('LINEABOVE',(0,0),(-1,0),.8,NAVY),('LINEBELOW',(0,0),(-1,0),.5,NAVY),
      ('LINEBELOW',(0,-1),(-1,-1),.6,NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafb')]),
      ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
    ]));story.extend([t,Spacer(1,9)])
def figure(name,caption,width=475,height=None):
    from PIL import Image as PILImage
    f=FIG/(name+'.png')
    if not f.exists():raise FileNotFoundError(f)
    im=PILImage.open(f);h=width*im.height/im.width
    if height and h>height:width=width*height/h;h=height
    story.extend([Image(str(f),width=width,height=h),Spacer(1,5),P(caption,'caption')])
def j(name):return json.loads((OUT/name).read_text())
def score(x):return f'{float(x):.5f}'

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self._pages=[]
    def showPage(self):self._pages.append(dict(self.__dict__));self._startPage()
    def save(self):
        n=len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self.setStrokeColor(colors.HexColor('#cfd7de'));self.setLineWidth(.4);self.line(54,42,A4[0]-54,42)
            self.setFont(BODY,8);self.setFillColor(colors.HexColor('#777777'))
            self.drawString(54,28,'边通书  |  House Prices 机器学习课程实验报告')
            self.drawRightString(A4[0]-54,28,f'{self._pageNumber} / {n}')
            super().showPage()
        super().save()

def main():
    stats=j('data_stats.json');valid=j('final_validation.json');branch=j('transaction_branch.json')
    submissions=j('kaggle_submissions.json');counts=j('experiment_counts.json');weights=j('optimized.json')
    done=sorted([r for r in submissions if r['status']=='complete' and r['publicScore']],key=lambda r:r['date'])
    best=min(done,key=lambda r:float(r['publicScore']));bestscore=float(best['publicScore'])
    improvement=(.12333-bestscore)/.12333*100
    allm=pd.read_csv(OUT/'all_experiments.csv');missing=pd.read_csv(OUT/'missingness.csv',index_col=0)
    corr=pd.read_csv(OUT/'correlations.csv',index_col=0).iloc[:,0]
    env=j('runtime_environment.json')
    story.append(Spacer(1,95));para('机器学习课程实验报告','title');story.append(Spacer(1,35))
    para('基于特征工程、残差学习与多模型集成的<br/>Ames 房价预测方法','cover');story.append(Spacer(1,42))
    for text in ['姓名：边通书','专业：计算机科学与技术','学号：26B351012','实验平台：Kaggle / 远程 Ubuntu 服务器','报告日期：2026 年 10 月 13 日']:
        para(text,'cover');story.append(Spacer(1,8))
    story.append(Spacer(1,35));para(f'最终已验证 Public Score：{bestscore:.5f}<br/>基线：0.12333；相对改善：{improvement:.2f}%','cover')
    para('成绩口径：官方 Kaggle Public Score，数值越小越好。','caption');page()

    section('1. 实验目的')
    para('本实验针对 Kaggle House Prices - Advanced Regression Techniques 比赛，依据住宅的地段、面积、质量、设施及交易属性预测销售价格 SalePrice。任务属于小样本表格回归。实验目标是在已有线上分数 0.12333 的基础上迭代模型，并争取将官方对数 RMSE 降到 0.1 以下。')
    para('实验重点包括：①完成数据探索与缺失语义分析；②在每个训练折内拟合填补、编码与缩放，降低验证信息泄露；③比较线性回归、核方法与多种提升树；④用 Optuna 搜索残差模型，并通过折外预测进行集成；⑤核验真实 Kaggle 分数；⑥分析异常交易、验证差异和模型局限，保存完整实验过程。')
    section('2. 数据描述')
    table(['数据文件','样本数','字段数','标签'],[['train.csv',1460,81,'包含 SalePrice'],['test.csv',1459,80,'SalePrice 未公开'],['sample_submission.csv',1459,2,'仅为提交格式示例']],[155,80,80,160])
    para('除标识列 Id 和目标列 SalePrice 外，共有 79 个解释变量：36 个数值字段与 43 个文本类别字段。数据描述文件给出了房屋质量、地下室、车库等属性的定义。Id 仅用于匹配提交，不作为预测特征；未使用外部房价标签或隐藏测试标签。')
    para(f'训练价格范围为 34,900 至 755,000 美元，中位数 163,000 美元，均值 180,921.20 美元。原价格偏度为 {stats["raw_skew"]:.3f}，自然对数价格偏度为 {stats["log_skew"]:.3f}。对数变换明显减弱长尾，使模型更重视相对误差。')
    figure('target_distribution','图 1  原始价格与自然对数价格分布。左图单位为千美元。',height=175)
    para('官方指标为预测价格和真实价格的自然对数差的 RMSE。新版本直接训练 z = ln(SalePrice)，预测后用 exp(z) 恢复价格。历史基线使用 log1p / expm1；该旧口径在本价格范围内与官方指标数值接近，但仍保留标识。','small')
    para('RMSE_log = sqrt((1/n) * sum((ln(y_i) - ln(p_i))^2))','formula');page()

    section('2.1 缺失值与变量语义')
    para('缺失值不一定表示信息未知。数据字典说明，Alley、PoolQC、Fence 等字段中的 NA 往往表示“无小巷、无泳池或无围栏”。这些类别保留为 None；不能因为缺失比例高就统一删除，也不应把 None 与未知值混为一种业务含义。')
    rows=[]
    for c in ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu','LotFrontage','GarageYrBlt','BsmtQual']:
        r=missing.loc[c];rows.append([c,int(r.train_count),f'{r.train_percent:.2f}%',int(r.test_count)])
    table(['字段','训练缺失数','训练缺失率','测试缺失数'],rows,[185,95,100,95])
    para('pandas 默认读取会将字面量 None 也视为缺失，因此 MasVnrType 的 872 个缺失不能直接解释为 872 个记录错误。此字段按无砖石贴面处理；其余结构性缺失与面积字段结合解释。LotFrontage 表示临街长度，其未知值按训练折内 Neighborhood 的中位数填补，再回退到训练折全局中位数。')
    figure('missingness','图 2  训练与测试的主要缺失比例。',height=225)
    para('训练集未发现除 Id 外的完全重复记录。上述统计用于描述数据；实际预处理的统计量均由当前训练折重新计算，不使用测试集或验证折的众数、中位数和缩放参数。','small');page()

    section('3. 方法描述')
    para('整体流程为：读取官方数据 → 按字段语义处理缺失 → 构造面积、年龄、质量与设施特征 → 在训练折拟合变换和编码 → 五折训练并保存折外预测 → 比较模型、搜索残差参数 → 约束集成权重 → 冻结参数复验 → 处理特殊交易 → 多随机种子预测 → 上传 Kaggle 核验。')
    section('4. 特征抽取与数据处理')
    sub('4.1 折内处理与一致的特征空间')
    para('每折分别拟合 LotFrontage 的分组中位数、其他数值中位数、未知类别的众数、One-Hot 类别集合、偏度筛选及缩放统计量。One-Hot 使用 handle_unknown=ignore，在部分实验中将低频类别合并；未知类别不会导致预测报错。训练与测试不再联合决定类别集合。')
    para('无地下室、无车库时相应面积和数量填 0；缺失 GarageYrBlt 用 YearBuilt 填补，明显晚于销售年份的车库年份回退到建造年份。住宅类型 MSSubClass 与月份 MoSold 转为类别；质量等级增加有序数值表示，并保留原类别供不同模型学习。')
    sub('4.2 面积异常与评价范围')
    para('继承旧基线的筛选条件：GrLivArea > 4000 且 SalePrice < 300000。该条件在官方训练集中定位到 Id 524 和 1299；其居住面积分别为 4676 和 5642 平方英尺，售价为 184750 和 160000 美元。两者均为 Partial 销售，明显偏离一般面积与价格趋势。')
    figure('area_quality','图 3  居住面积、销售价格及总体质量关系；红点为两条异常记录。',height=205)
    para('最初模型比较在其余 1458 行上进行，因此该分数只描述“常规样本”表现。后续另做全部 1460 行的验证：异常记录保留在验证折参与评分，只从普通模型的训练折排除。两种口径分别列出，避免把删去难样本后的分数解释成全体泛化性能。','small');page()

    section('4.3 面积、年龄与质量特征')
    rows=[
      ['TotalSF','地下室总面积 + 一楼面积 + 二楼面积','联合反映总建筑规模'],
      ['FinishedSF','地上居住面积 + 两类已装修地下室面积','区分可使用面积与毛坯面积'],
      ['EffectiveSF','一楼 + 0.8×二楼 + 0.5×已装修地下室 + 0.3×车库','先验加权面积；权重未经声称为真实市场系数'],
      ['TotalBath','全卫 + 0.5×半卫 + 地下室全卫 + 0.5×地下室半卫','综合卫生设施'],
      ['TotalPorchSF','露台、门廊及相关室外面积之和','衡量附属空间'],
      ['HouseAge / RemodAge','销售年份 - 建造或翻修年份，负值截为 0','房龄和翻修新旧程度'],
      ['QualArea / QualTotalSF','OverallQual × 居住面积或 TotalSF','表达质量与规模的交互'],
      ['HasGarage / HasBsmt 等','相应面积或数量是否大于 0','分离设施存在性与大小'],
      ['扩展交互','质量×log1p(面积)、质量×log1p(房龄)、社区与房型组合','检验额外非线性及地段交互'],
    ]
    table(['特征','定义','作用'],rows,[125,205,145],small=True)
    sub('4.4 分布变换与缩放')
    para('针对训练折中非负、非二元且绝对偏度大于 0.75 的数值特征，对比 log1p 与 Box-Cox（固定 λ=0.15）变换。线性与核模型对比 StandardScaler 和 RobustScaler；后者按中位数和四分位距缩放。本实验中 RobustScaler 的 Lasso 明显更好，说明稀疏类别与长尾特征的缩放方式会影响正则化和核距离。')
    para('BoxCox1p(x; lambda) = ((1 + x)^lambda - 1) / lambda','formula')
    para(f'原始价格 Pearson 相关性中，OverallQual 为 {corr["OverallQual"]:.3f}，GrLivArea 为 {corr["GrLivArea"]:.3f}。这些是描述性相关，不能解释为因果效应。地下室、一楼面积等变量间存在共线性，因此同时比较正则化回归和树模型。')
    figure('feature_importance','图 4  CatBoost 四层模型五折平均特征重要性。此指标解释模型依赖，不代表因果效应。',height=180);page()

    section('5. 回归模型设计')
    table(['模型家族','建模思路','主要参数 / 实验设置'],[
      ['Ridge / Lasso / ElasticNet','线性回归 + L2、L1 或混合正则化','Ridge α=5/15/35；Lasso α=0.0003/0.0005/0.0008'],
      ['SVR-RBF','用核方法学习平滑非线性','C=10 至 50；ε=0.008；γ=0.00015 至 0.0005'],
      ['Kernel Ridge','核化的 L2 回归','二次多项式核，α=0.6'],
      ['Random Forest','随机采样与多树平均','600 棵树；叶节点至少 2 个样本'],
      ['GBR-Huber','梯度提升与稳健损失','3000 棵树；深度 3；学习率 0.025'],
      ['LightGBM','叶节点方式构建提升树','叶数 5/8/12/20；早停 200 轮'],
      ['XGBoost','正则化梯度提升树','深度 2/3/4；GPU hist；早停 200 轮'],
      ['CatBoost','原生类别处理与有序提升','CPU Ordered 深度 4/5/6；另测 GPU Plain 深度 5'],
      ['Lasso + residual XGB','先拟合整体趋势，再学习价格对数残差','Optuna 联合搜索基础 Lasso 与残差树参数'],
    ],[115,175,185],small=True)
    sub('5.1 残差学习与集成')
    para('Lasso 用 L1 正则化约束回归系数，适合编码后具有相关和冗余特征的设计矩阵。SVR 的 RBF 核把距离映射为相似度，参数 C 控制误差惩罚、γ 控制核的作用范围，ε 给出不惩罚的小误差带。两者分别提供线性主趋势和平滑非线性。')
    para('Lasso: min_beta (1/(2n)) * ||z - X beta||_2^2 + alpha * ||beta||_1','formula')
    para("RBF kernel: K(x, x') = exp(-gamma * ||x - x'||_2^2)",'formula')
    para('残差模型先在训练折拟合 Lasso，得到 b(x)，再用 XGBoost 学习 r = ln(y) - b(x)，最终预测为 b(x) + g(x)。其训练残差来自同一训练折的拟合；验证标签只用于常规早停，不进入训练目标。复验阶段固定训练轮数，停止依靠当前验证标签选轮数。')
    para('z_hat(x) = b(x) + g(x);    z_blend(x) = sum(w_m * z_m(x))','formula')
    para('按模型家族选代表模型，利用折外预测最小化平方误差确定非负集成权重，约束权重之和为 1、每项不超过 0.55，并加小幅均匀权重收缩项。输出价格为 exp(z_blend)，等价于在价格空间做加权几何平均。')
    para('另对权重学习做五折交叉拟合，但基础模型和参数曾在这批样本上选择，因此该结果是选择偏差诊断，不是完全独立的嵌套验证分数。','small');page()

    section('6. 实验设置及结果')
    sub('6.1 环境、随机性与验证协议')
    para(f'服务器为 Ubuntu，提供 128 个 CPU 逻辑核与 8 张 A100 40GB。使用已有 Python 3.8 环境，每个训练任务最多 8 个 CPU 线程，GPU 任务使用当时空闲的 4、5 号设备。各任务分批执行；未修改其他用户的进程。')
    table(['关键库','版本'],[[k,env['packages'][k]] for k in ['scikit-learn','xgboost','lightgbm','catboost','optuna']],[250,225])
    para(f'最终完成 {counts["five_fold_configurations"]} 组统一五折配置，即 {counts["five_fold_model_fits"]} 次模型拟合；另做 40 次 Optuna 搜索和 40 次固定配置复验拟合。主五折为 KFold(shuffle=True, random_state=42)，Optuna 三折搜索种子 631，新划分复验种子 2026，完整样本审计种子 73。')
    para('Optuna 使用 TPE 与中位数剪枝，搜索最大深度、学习率、采样率、列采样、L1/L2 正则项、最小叶权重、特征方案及 Lasso 强度。挑选最优四组配置后重新跑统一五折，不能把三折最优搜索分数与五折比较表直接混用。')
    table(['评价范围 / 协议','OOF 对数 RMSE','样本数'],[
      ['历史基线（log1p；旧预处理）','0.11123',1458],
      ['初步集成 / 新版折内预处理',score(j('preliminary.json')['oof_rmse']),1458],
      ['残差集成 / 参数选择划分',score(valid['selection']['rmse']),1458],
      ['权重交叉拟合诊断',score(weights['meta_crossfit_rmse']),1458],
      ['固定权重与训练轮数 / 新种子',score(valid['confirmation']['rmse']),1458],
      ['完整样本 / 普通模型',score(valid['all_rows_audit']['rmse']),1460],
      ['完整样本 / 增加交易分支',score(branch['all_rows_branch_rmse']),1460],
    ],[290,115,70])
    para('不同种子复验仍复用相同训练标签，不是独立测试集；异常交易分支也在探索后提出。表中明确标注选择、复验和审计口径。各模型原始逐折分数及耗时可查 experiments/*/fold_*/metrics.json。','small');page()

    section('6.2 模型与特征消融对比')
    wanted=[('原始字段 + Ridge','raw_ridge'),('扩展特征 + Ridge','ridge_extended_20'),('原始字段 + Lasso / Robust','lasso_raw_boxcox_robust'),('扩展特征 + Lasso / Robust','lasso_extended_boxcox_robust'),('扩展特征 + Lasso / Standard','lasso_extended_boxcox_standard'),('扩展特征 + SVR','svr_extended_10_0.0005'),('XGBoost 深度 3','xgb_d3'),('最优残差 XGBoost','optuna_23'),('LightGBM 8 叶','lgb_8'),('CatBoost CPU 深度 4','cat_cpu_d4'),('CatBoost GPU 深度 5','cat_gpu_d5'),('GBR-Huber','gbr_huber'),('Random Forest','random_forest')]
    rows=[]
    for label,name in wanted:
        r=allm.loc[allm.name==name].iloc[0];rows.append([label,score(r.rmse),score(r.fold_std)])
    table(['方法','OOF RMSE','折间标准差'],rows,[270,100,105])
    para('扩展特征使原始 Lasso 从 0.11018 改善到 0.10906；但该比较包含额外交互，不能把全部收益归给某一个面积指标。StandardScaler 下扩展 Lasso 为 0.12189，明显差于 RobustScaler。残差模型为 0.10613，较纯 XGBoost 的 0.11241 更好，说明先学习平滑主趋势有帮助。')
    para('随机森林在本数据上为 0.13004，较多浅层或正则化模型差；CatBoost 最优单模型也没有优于 SVR。GPU CatBoost 未得到更优分数，且不同 CPU/GPU 实验的提升方式不同，不能将其耗时差视为严格的硬件基准。')
    figure('model_comparison','图 5  参数选择划分上的代表实验与集成分数。图中虚线为选择后的集成结果。',height=190);page()

    section('6.3 实际 Kaggle 提交与稳定性')
    descriptions={'submission.csv':'v1 旧版 XGB + Lasso + LGB','submission_preliminary.csv':'v2 折内特征 + SVR 集成','submission_optimized.csv':'v3 Optuna 残差集成','submission_multiseed.csv':'v4 多种子 / 固定配置','submission_transaction_branch.csv':'v5 增加特殊交易分支','submission_fullfit_bagged.csv':'v6 全量拟合与折模型混合'}
    rows=[]
    for r in done:
        date=datetime.fromisoformat(r['date'].replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8)))
        rows.append([descriptions.get(r['fileName'],r['fileName']),score(r['publicScore']),date.strftime('%H:%M'),r['ref']])
    table(['提交方法','Public Score','北京时间','提交编号'],rows,[235,85,65,90],small=True)
    figure('kaggle_iterations','图 6  官方 API 返回的实际 Public Score。下降表示改善；虚线为 0.1 目标。',height=185)
    para(f'当前最佳线上分数为 {bestscore:.5f}，较初始 0.12333 相对改善 {improvement:.2f}%。'+('已低于 0.1 目标。' if bestscore<.1 else '仍未低于 0.1 目标。')+' Private Score 未返回，本报告不推测最终私榜排名。全部线上记录保存在 output/kaggle_submissions.json。')
    para('v3 的本地选择分数优于 v2，但线上 0.12285 差于 v2 的 0.12204；v4 多种子平均也仅为 0.12279。这说明更多搜索或更低选择分数并不保证更好的测试表现。v5 在与 v4 相同的普通预测上仅修改特征条件触发的特殊交易分支，线上改善到 0.11477。')
    fullfit=[r for r in done if r['fileName']=='submission_fullfit_bagged.csv']
    if fullfit:para(f'v6 在 1458 条常规训练记录上拟合 12 个可重新加载的最终管线，并与折模型预测平均，保留相同交易分支。线上为 {float(fullfit[0]["publicScore"]):.5f}，没有超过 v5，因此默认提交保留 v5。该候选没有独立 OOF 分数，训练误差不列入验证比较。')
    para('公开榜只覆盖测试集的一部分；在其上选择最优版本也会产生适应性选择偏差，因此其成绩是比赛提交表现，不能外推为未知城市房价的无偏泛化误差。','small');page()

    section('7. 实验结果分析')
    sub('7.1 特殊交易对误差的影响')
    para('普通模型把两条大面积、低价 Partial 交易当作正常高端房产：完整样本审计中分别预测约 826867 与 1308154 美元，而真实价格为 184750 与 160000 美元。仅两条记录便使整体 RMSE 从常规样本约 0.104 上升到 0.12417。测试中也存在同样的大面积 Partial 条件，但其真实标签始终未知。')
    para('定义特征分支：GrLivArea > 4000 且 SaleCondition = Partial。阈值 4000 源自已有基线面积筛选，分支不依赖测试 Id。普通模型仍负责其他住宅；该分支用当前训练折同类交易的对数均值预测，即同类价格的几何均值。')
    para('p_expert = exp(mean(ln(SalePrice_large_partial_train)))','formula')
    table(['留出验证对象','该折专家训练对象','预测价格','真实价格'],[['Id 1299','仅 Id 524','184750','160000'],['Id 524','仅 Id 1299','160000','184750']],[120,135,110,110])
    para(f'两次预测均未使用对应验证对象的价格。全训练集拟合后分支价格为 {branch["fitted_price"]:.2f} 美元；触发测试记录数为 1。完整 1460 行的分支 OOF RMSE 为 {branch["all_rows_branch_rmse"]:.5f}，较普通完整样本审计显著下降。')
    table(['面积阈值（平方英尺）','训练触发数','测试触发数','分支拟合价格'],[[s['threshold'],s['train_n'],s['test_n'],f'{s["estimated_price"]:.2f}'] for s in branch['threshold_sensitivity']],[175,85,85,130])
    para('3800 至 4500 的阈值触发同样的两条训练和一条测试记录，说明结果不依赖某个极窄的切点。但专家仅有两个样本，Partial 也不能直接等同于“未完工售价”；分支是根据本数据提出的建模假设，不能作为一般估价业务规则。其验证是在探索之后实施，仍存在选择偏差。','small');page()

    sub('7.2 残差、地段与极端误差')
    figure('final_residuals','图 7  含特殊交易分支的完整样本折外预测与残差。',height=180)
    para('总体预测沿对角线分布，残差多集中在 0 附近；仍有少数房产出现较大负残差，表示模型高估。除特殊大面积记录外，家族或非正常销售、房屋实际状况及未记录交易因素可能使属性接近的房屋售价相差较大。这些原因属于解释性假设，字段本身不能验证全部交易背景。')
    errs=pd.read_csv(OUT/'largest_errors.csv').head(5)
    table(['Id','真实价格','预测价格','销售条件','对数残差'],[[int(r.Id),f'{r.actual_price:.0f}',f'{r.prediction_price:.0f}',r.SaleCondition,f'{r.residual_log:.3f}'] for _,r in errs.iterrows()],[45,95,95,135,105])
    para(f'常规样本的两种子 OOF 预测中，误差最大的 10 个样本贡献约 {valid["top10_squared_error_fraction"]*100:.2f}% 的总平方误差。此占比反映 RMSE 对极端误差的敏感性，不能据此继续删除这些困难样本以追求更低验证成绩。')
    group=pd.read_csv(OUT/'all_rows_errors_by_SaleCondition.csv')
    table(['SaleCondition','样本数','普通模型完整样本 RMSE','平均对数残差'],[[r.group,int(r.n),score(r.rmse),f'{r.mean_residual:.4f}'] for _,r in group.iterrows()],[140,65,170,100])
    para('表中按销售条件分组的是交易分支之前的完整样本审计，用于定位误差，不是分支后的性能表。小组只有少数样本时，分数波动很大；不以单组优劣判断模型对所有同类住宅的可靠性。','small');page()

    section('7.3 优势、局限与后续方向')
    para('有效改进来自三个方面：首先，面积、房龄、质量交互与稳健缩放更贴近小样本表格回归；其次，Lasso 的平滑主趋势与残差树、SVR 的不同误差结构具有互补性；最后，完整样本审计发现了被常规异常值删除掩盖的特殊交易误差。线上收益主要来自对这种数据机制的处理，而非一味扩大模型复杂度。')
    para('局限包括：①训练样本仅 1460 条，稀有条件下无法可靠估计价格机制；②选择过多配置会过拟合验证划分，权重交叉拟合不能完全消除基础模型选择偏差；③新随机种子仍复用训练标签，不等于独立外部验证；④公开榜未提供所有测试样本标签，私榜分数与排名未知；⑤特征重要性和价格相关性不代表因果关系。')
    para('校准也进行了对照：一、二、三次多项式的额外交叉拟合分数约为 0.10355、0.10375、0.10370，相比两种子 OOF 0.10357 改善极小。因此保存校准候选用于分析，没有把它当作主要有效改进。未经真实验证的候选不会在结果表中获得虚构线上分数。')
    para('后续应先固定一套较少的候选与验证协议，使用嵌套验证或明确保留的评估样本，再检查高误差交易和罕见类别的机制。若新增数据，应仅使用允许的独立辅助特征，并说明来源；不从公开原始数据反查隐藏测试标签。')
    section('8. 总结与复现说明')
    para(f'本实验实现了从数据探索、折内预处理、模型比较、Optuna 搜索、残差集成到线上评分核验的完整流程。当前最佳实际 Public Score 为 {bestscore:.5f}，相对基线改善 {improvement:.2f}%。'+('目标已达到。' if bestscore<.1 else '0.1 以下的目标尚未达到；本报告保留真实成绩，不以常规样本验证分数替代线上评分。'))
    table(['文件 / 目录','作用'],[
      ['experiment.py / optimize.py','五折实验、折内预处理和 Optuna 搜索'],
      ['analyze.py / finalize.py','探索图、受约束权重、稳定性验证与误差分组'],
      ['transaction_branch.py','只用官方训练标签拟合特殊交易分支'],
      ['train_final.py / models/','保存全训练集管线，重新加载并检验预测一致性'],
      ['experiments/ / output/','逐折参数、OOF、耗时、CSV、图与评分记录'],
      ['submission.csv','实际已验证最佳版本；历史基线已备份'],
    ],[185,290],small=True)
    para('服务器路径：/mnt/data/bts/repos/homework/ml。执行方法见 README.md。训练代码与报告源中不含 SSH 密码或 Kaggle Token；认证通过进程环境提供。数据哈希、随机种子与软件版本均已保留。','small');page()

    section('附录 A：全部五折配置结果')
    para('以下为种子 42、常规 1458 行口径的全部配置。OOF RMSE 是汇总每条记录的折外误差；折间标准差不是统计置信区间。重复或相近配置作为搜索轨迹保留。Optuna 三折原始试验及剪枝状态另见 experiments/optuna/trials.csv。','small')
    rows=[[r['name'],score(r.rmse),score(r.fold_std)] for _,r in allm.iterrows()]
    table(['配置名称','OOF RMSE','折间标准差'],rows[:28],[300,85,90],small=True);page()
    section('附录 A：全部五折配置结果（续）')
    table(['配置名称','OOF RMSE','折间标准差'],rows[28:],[300,85,90],small=True)
    para('实验命名中的 raw / engineered / extended 分别表示原字段、面积年龄等工程特征、以及扩展交互；log / boxcox 为分布变换；robust / standard 为缩放。完整超参数由每个模型目录的 metrics.json 和各 run_config.json 记录。','small');page()

    section('附录 B：最终普通集成参数与参考资料')
    cfg=json.loads((ROOT/'frozen_config.json').read_text());fixed=json.loads((ROOT/'fixed_iterations.json').read_text())
    for s in cfg:
        sub(s['name'])
        para(f'权重：{weights["weights"][s["name"]]:.6f}；特征：{s["level"]}；变换：{s["transform"]}；稀有类别最小频数：{s.get("rare",1)}。','small')
        params={k:v for k,v in s['params'].items() if k not in ['device','devices']}
        para('参数：'+escape(', '.join(f'{k}={v}' for k,v in params.items()))+'。','small')
        if s['name'] in fixed:para(f'新种子复验固定训练轮数：{fixed[s["name"]]}；选自参数选择划分各折早停轮数的中位数。','small')
    sub('参考资料与数据依据')
    refs=[
      ('[1] Kaggle. House Prices - Advanced Regression Techniques. 比赛概览、数据与评价指标。','https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques'),
      ('[2] Kaggle 官方下载数据及 data_description.txt。所有统计由本实验输入文件计算。','https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data'),
      ('[3] Scikit-learn. Preprocessing、linear models、SVR 与 model selection 文档。','https://scikit-learn.org/stable/user_guide.html'),
      ('[4] XGBoost. Tree booster、正则化与 GPU 参数文档。','https://xgboost.readthedocs.io/en/stable/parameter.html'),
      ('[5] CatBoost. 参数与类别特征处理文档。','https://catboost.ai/docs/en/'),
      ('[6] Optuna. TPE、Pruner 与可复现参数搜索文档。','https://optuna.readthedocs.io/en/stable/'),
    ]
    for text,url in refs:para(escape(text)+f'<br/><link href="{url}" color="#326080">{url}</link>','small')
    para('报告章节与封面形式参考用户提供的《示例报告.pdf》，实验内容、图表和结果均为本次房价回归项目的数据与计算结果。线上分数核验日期为 2026 年 10 月 13 日。','small')
    path=DEST/'House_Prices_机器学习实验报告_边通书.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=58,
                         title='基于特征工程、残差学习与多模型集成的 Ames 房价预测方法',author='边通书')
    doc.build(story,canvasmaker=NumberedCanvas)
    print(path)

if __name__=='__main__':main()
