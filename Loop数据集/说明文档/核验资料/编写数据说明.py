"""Read original headers and official documentation; write documentation only."""
from pathlib import Path
import json,re
P=Path(__file__).resolve().parent
BASE=P.parents[1]; RAW=BASE/'原始数据'; DOC=P.parent
A=json.loads((P/'官方字典展开.json').read_text())
CN={}
def mapping(text):
 for line in text.strip().splitlines():
  k,v=line.split('=',1);CN[k]=v
mapping('''PtID=去标识化参与者编号；实际表头存在，官方多数表的字典未单列说明
RecID=本表内唯一记录编号；不能当作跨表患者编号
ParentLOOPDeviceUploadsID=所属上传记录的编号，对应 LOOPDeviceUploads.RecID
DeviceDtTm=设备本地日期时间；字典明确多数 Tidepool 记录不提供该字段
UTCDtTm=与时区偏移关联/调整的事件日期时间；官方不同表措辞略有不同，不能再次盲目加减时差
TmZnOffset=时区偏移；官方字典未注明单位和正负号规则
OriginName=记录来源名称
OriginVers=记录来源版本
OriginType=记录来源类型
OriginDeviceFirmwrVer=来源设备固件版本
OriginDeviceHardwrVer=来源设备硬件版本
OriginDeviceManufact=来源设备制造商
OriginDeviceModel=来源设备型号
OriginOperatingSystVer=来源操作系统版本
OriginProductType=来源产品类型
ContactDt=研究人员与参与者联系日期
ContactInitBy=联系由谁发起：参与者或 Jaeb 研究人员
ContactType=联系方式/联系记录类型；并非对话正文
RecordSubType=血糖仪记录子类型；字典没有列出全部枚举
BGMVal=血糖仪的点测血糖读数；单位看 Units
Units=该记录数值的单位；不同表不保证相同
BasalType=基础输注类型；具体类型须与记录来源一起解释
Duration=输注持续时间；基础率表为毫秒，Bolus 表按数据来源区分，见本表补充
ExpectedDuration=预期输注持续时间；不是已执行时长，单位见本表补充
Percnt=应按被覆盖基础率的多大百分比输注；字典未给数值尺度，不能先假定 0–1 或 0–100
Rate=此基础事件的输注速率，单位 U/h；不是一次注射的 U 数
SuprBasalType=因当前基础事件生效而被覆盖、未执行的基础输注类型
SuprDuration=被覆盖基础事件的时长；字典未单独注明单位
SuprRate=被覆盖的基础输注速率；不应与当前 Rate 相加，字典未单独注明单位
BolusType=大剂量输注子类型，例如普通或方波；不是餐食/纠正目的的可靠标签
Normal=普通部分的记录输注量，U
ExpectedNormal=普通部分的预期输注量，U；不等于已执行量
Extended=延长输注部分的记录量，U
ExpectedExtended=延长部分的预期输注量，U；不等于已执行量
RecordType=记录类别；同一个文件中可以有 CGM 和 Calibration 校准记录
CGMVal=CGM 血糖读数；字典明确 Tidepool 值为 mmol/L，同时保留 Units
DexInternalDtTm=Dexcom 内部日期时间；不能直接取代事件时刻
DexTrend=Dexcom 趋势信息；字典未给出编码/斜率换算表
ExerciseName=运动名称/类别
DistanceValue=运动距离数值，配合 DistanceUnits
DistanceUnits=运动距离单位
DurationValue=运动持续时间数值，配合 DurationUnits
DurationUnits=运动持续时间单位
EnergyValue=运动能量消耗数值，配合 EnergyUnits
EnergyUnits=运动能量单位
ReportedIntensity=上报运动强度；字典未定义统一分级
CarbsNet=记录的净碳水化合物量，配合 CarbUnits；不能保证等于实际进食量
CarbUnits=净碳水量的单位
RptGeneratedDt=设备问题报告生成日期；不是每个配置开始生效的精确时刻
LoopVers=Loop 软件版本
PumpModel=胰岛素泵型号
PumpManager=泵管理器信息；字典没有进一步解释内部结构
RileyLinkFirmwrVers=RileyLink 通信设备固件版本
CarbRatioSched=不同时段的碳水/胰岛素比设置；保留原始配置，字典未逐项说明结构/单位
DefAbsorptTimeFast=快速吸收的默认时间设置；字典未注明单位
DefAbsorptTimeMed=中速吸收的默认时间设置；字典未注明单位
DefAbsorptTimeSlow=慢速吸收的默认时间设置；字典未注明单位
InsulinSensSched=不同时段的胰岛素敏感性设置；并非直接测得的生理真值
InsulinModel=当时 Loop 使用的胰岛素作用模型配置；不是本项目训练的模型
BasalProfile=设定的基础率时间表；不是实际执行的基础输注轨迹
MaxBasalRate=设备配置中的最大基础率；不是本项目已经验证的安全上限，字典未逐项注明单位
MaxBolus=设备配置中的最大 Bolus；不是本项目已经验证的安全上限，字典未逐项注明单位
RetroCorrectEnabled=是否启用回顾性校正的设置；具体布尔编码未在字典列出
SuspThreshold=暂停输注的血糖阈值设置；字典未注明单位
OverrideRangePremealMin=餐前覆盖目标范围下限；字典未注明单位
OverrideRangePremealMax=餐前覆盖目标范围上限；字典未注明单位
OverrideRangeWorkoutMin=运动覆盖目标范围下限；字典未注明单位
OverrideRangeWorkoutMax=运动覆盖目标范围上限；字典未注明单位
LoopSettings=Loop 设置的整体内容；不能当作已规范化数值矩阵
DeviceManufact=上传设备的制造商，可包含多个
DeviceModel=上传设备型号
DeviceType=上传设备类型，例如 CGM、BGM
Timezone=上传者选择的时区；不能自动认定等于事件发生地点
DatasetType=上传数据集的类型；不是机器学习训练/测试划分
UploadComputerDtTm=上传到 Tidepool 时的电脑时间
UploadUTCDtTm=上传到 Tidepool 时的 UTC 时间；不是所有记录的事件时间
UploadClientVer=Tidepool 上传软件版本
ClientName=客户端名称
ClientVersion=客户端版本
DataSource=数据来源，例如 Tidepool、Diasend
RecommendedCarb=计算器推荐用于覆盖碳水的胰岛素量，U；推荐不等于已输注
RecommendedCorrection=计算器推荐用于纠正血糖的胰岛素量，U
RecommendedNet=计算器的净推荐结果；字典未单独解释单位/合成规则
BgInput=输入计算器的血糖；字典原文写 mg，单位不完整，尚不能直接当 mg/dL
CarbInput=输入计算器的碳水；字典原文写 mg，存在疑点，不能擅自改成 g
InsulinOnBoard=计算器估计的体内活性胰岛素，U；属于模型估计值
InsulinCarbRatio=计算器使用的胰岛素/碳水比；字典没有充分定义方向和单位
InsulinSensitivity=单位胰岛素对应的血糖变化参数；血糖单位需另核
BgTargetLow=计算器血糖目标下限；字典未注明单位
BgTargetHigh=计算器血糖目标上限；字典未注明单位
BgTargetTarget=计算器目标血糖值；字典未注明单位
BgTargetRange=计算器目标范围参数；字典未细化表达方式和单位
BolusRecID=字段名提示可能关联 Bolus.RecID；官方释义为空，关联完整性尚未验证
WithdrawDt=退出研究的日期
WithdrawBy=由参与者或 Jaeb 发起退出
WithdrawReas=退出研究的原因
WithdrawLastSurv=是否愿意完成最后一组问卷
EnrollDt=入组日期
VisitSchedStartDt=随访计划计算的起始日期
PtStatus=参与状态：Active 进行中、Completed 完成、Dropped 退出
PtCohort=研究队列：A 新使用 Loop，B 已在使用 Loop
AgeAtEnrollment=入组年龄
LoopUseTimeAtEnroll=入组时已使用 Loop 的时长类别
PtTimezoneOffset=参与者时区偏移；字典未注明单位和符号规则
PrefContactMethod=偏好联系方式：邮件、短信或两者
Visit=访视/随访阶段
CollectionDt=检验样本采集日期
AnalysisDt=样本分析日期；与采样日期区分
ResultName=检验项目名称
Value=检验结果数值，结合项目和 Units
SubjectID=去标识化参与者编号；不同表实际拼写大小写不同
subjectID=去标识化参与者编号；对应患者身份，不是事件编号
Period=研究阶段/统计区间标签；不是逐条事件时间戳
ageAtBaseline=基线入组年龄
age_diabetes_developed=发生糖尿病时的年龄
age_first_used_pump=首次使用胰岛素泵的年龄
age_started_cgm=首次使用 CGM 的年龄
gluInRange=70–180 mg/dL 范围内时间比例，%
gluMean=该阶段平均血糖，mg/dL；与原始 CGM 的 mmol/L 区分
gluAbove180=高于 180 mg/dL 的时间比例，%
gluAbove250=高于 250 mg/dL 的时间比例，%
gluAUC180=以 180 mg/dL 为阈值的曲线下面积指标；具体归一化/计算公式字典未提供
gluHBGI=高血糖风险指数；字典未提供计算公式
gluBelow70=低于 70 mg/dL 的时间比例，%
gluBelow54=低于 54 mg/dL 的时间比例，%
gluLBGI=低血糖风险指数；字典未提供计算公式
gluAOC70=以 70 mg/dL 为阈值的曲线上方面积指标；具体归一化/计算公式字典未提供
gluSD=阶段血糖标准差，mg/dL
gluCV=阶段血糖变异系数，%
gluHypo54Rate=每周 CGM 低血糖事件率；字段名含54，字典未给持续时间/事件合并规则
gluHours=该统计阶段实际提供 CGM 数据的小时数
totalDailyIns=该阶段平均每日实际输注总量，U/日
totalDailyInsBasal=该阶段平均每日基础输注量，U/日
totalDailyInsBolus=该阶段平均每日 Bolus 输注量，U/日
totalDailyInsPerKg=按体重折算的平均每日总量，U/kg/日
totalDailyInsBasalPerKg=按体重折算的平均每日基础量，U/kg/日
totalDailyInsBolusPerKg=按体重折算的平均每日 Bolus 量，U/kg/日
event=上报不良事件类型；不是从 CGM 自动检出的全部事件
eventDt=上报不良事件发生日期
date=设备问题上报日期
issue=设备问题类型
device=受到影响的设备''')
# Questionnaire meanings; exact official wording and answer codes are retained separately.
mapping('''gender=参与者性别（官方题目 gender）
ethnicity=是否为西班牙裔/拉丁裔
race=自报种族类别
race_multiple=多种族的自由文字补充
pregnant=当前是否怀孕；特别注意 2=否，不是0
height_feet=身高的英尺部分；不是总身高厘米值
height_inches=身高的英寸部分；与英尺部分配套
weight=自报体重，磅（lbs），不是 kg
duration=T1D 病程；字典未单列单位，不能与设备 Duration 混同
hba1c_when_measured=距最近一次 HbA1c 测量的时长类别
hba1c_level=自报最近 HbA1c 水平；不同于 SampleResults 的实验室记录
low_carb_diet=是否遵循低碳水饮食
pump_use=使用胰岛素泵的经历/当前使用状态
pump_use_length=已经使用胰岛素泵的时长类别
insulin_type=泵内使用的胰岛素制剂类别
insertion_length=通常多少天更换输注套件
insulin_injection=是否在用泵之外还规律注射胰岛素；不是逐次注射剂量日志
afrezza=是否在用泵之外还规律使用 Afrezza 吸入胰岛素
how_calculate_bolus=餐时通常如何决定 Bolus 量
cgm_experience=CGM 使用经历/当前使用状态
cgm_use_length=CGM 使用时长类别
typical_cgm_location=CGM 传感器通常放置位置
aid_use=是否用过其他自动胰岛素输注系统
pump_type_use=Loop 配套泵型号
what_cgm=Loop 配套 CGM 型号
lifetime_hypo_events=自诊断 T1D 以来的严重低血糖事件自报次数
months_hypo_events=过去3个月严重低血糖事件自报次数
lifetime_dka_events=自诊断 T1D 以来的酮症酸中毒自报次数
months_dka_events=过去3个月酮症酸中毒自报次数
aware_low_blood=开始发生低血糖时的感知程度：0从不感知，5始终感知
how_low_blood=开始感觉低血糖症状时的血糖范围/从无症状
 describe_health=总体健康自评
activity_tracker=使用/计划使用的活动追踪设备；不是该设备连续数据本身
exercise_30_min=典型一周中每天至少运动30分钟的天数
aerobic_exercise=典型一周中有氧运动天数
anaerobic_exercise=典型一周中无氧运动天数
loop_version=所用 Loop 版本的枚举编码；数字不等于版本号
loop_branch=所用 Loop 软件分支
 typically_use_loop=Loop 典型使用时段：全天、仅夜间、部分日期
loop_apple_watch=是否使用 Loop Apple Watch App
hcp_interaction_loop=医护人员对本人使用 Loop 的知情/支持情况
use_insulin_pump=随访时是否仍使用胰岛素泵
cgm_use=随访时是否使用 CGM
not_skld_mang_dia_shld=因自觉糖尿病管理能力不足而感到困扰的程度
not_much_insul_shld=因自觉未使用应有胰岛素量而感到困扰的程度
 dnt_chk_bg_oft_shld=因自觉血糖检查不够频繁而感到困扰的程度
dnt_give_much_atten_shld=因自觉关注糖尿病不够而感到困扰的程度
tech_made_life_better=认同糖尿病技术改善了生活的程度
tech_made_life_easier=认同糖尿病技术让生活更轻松的程度
tech_made_health_better=认同糖尿病技术改善健康的程度
tech_more_good_than_bad=认同糖尿病技术利大于弊的程度
tech_more_time_work_than_worth=认同技术耗时费力超过其价值的程度；题意与前四项相反
bed_time=过去一个月通常的晚间上床时间
min_to_fall_sleep=过去一个月通常每晚入睡所需分钟数
wake_up_time=过去一个月通常的早晨起床时间
hours_sleep=过去一个月通常每晚实际睡眠小时数；不是卧床时长
cant_sleep_in_30_min=过去一个月因30分钟内不能入睡而出现睡眠困难的频率
wake_up_middle_night=过去一个月半夜/清晨醒来的频率
use_bathrm_night=过去一个月夜间起床如厕的频率
cant_breathe=过去一个月呼吸不畅影响睡眠的频率
cough_snore_loudly=过去一个月咳嗽/大声打鼾影响睡眠的频率
too_cold=过去一个月太冷影响睡眠的频率
too_hot=过去一个月太热影响睡眠的频率
bad_dreams=过去一个月噩梦影响睡眠的频率
have_pain=过去一个月疼痛影响睡眠的频率
rate_overall_sleep_quality=过去一个月总体睡眠质量：字典1很差至4很好；勿套用其他版本PSQI编码
take_sleep_meds=过去一个月服用助眠药物的频率
trouble_staying_awake=过去一个月驾车、吃饭、社交时难以保持清醒的频率
keep_enough_enthusiasm=过去一个月做事难以保持热情的问题程度
when_exrcse=对运动时应对低血糖的信心
when_sleep=对睡眠时应对低血糖的信心
when_drive=对驾车时应对低血糖的信心
when_soc_sit=对社交时应对低血糖的信心
when_alne=对独处时应对低血糖的信心
avd_prob_hypo=对避免低血糖导致严重问题的信心
catch_respnd=对血糖降得太低之前识别并应对的信心
cont_despite_hypo=对尽管有低血糖风险仍能从事想做活动的信心
confdt_spouse=估计伴侣对本人避免低血糖严重问题的信心；不是伴侣直接作答
pump_graphs_averg=利用血糖仪/泵图表或平均值发现偏高偏低的频率
search_online=上网搜索血糖问题信息的频率
text_another_person=将血糖发短信给他人求助的频率
talk_people_online=在线与其他糖尿病患者交流求助的频率
calculate_bolus=使用计算器/App/泵计算器计算胰岛素量的频率
carb_counter_app_web=使用App/网站估算食物碳水的频率
set_alarms=设置糖尿病相关事务提醒的频率
see_graphs_after_selfcare=改善自我管理后查看血糖图表/日志评估效果的频率
contact_clinic_email_website=通过邮件或患者网站联系诊所求助的频率
general_risk=总体冒险意愿，0完全不愿至10非常愿意
finacial_risk=财务冒险意愿；原始字段拼写 finacial 保留
career_risk=职业冒险意愿
health_risk=健康相关冒险意愿
nervous_about_loop=开始使用 Loop 时的紧张程度
trust_loop=是否信任 Loop 正常工作
number_weeks_to_trust_loop=建立对 Loop 正常工作的信任所需周数
recommend_loop=向其他 T1D 患者推荐 Loop 的可能性
not_use_loop=是否认为某类 T1D 患者不适合用 Loop；不是本人是否停用
hard_start_loop=开始使用 Loop 的困难程度
help_starting_loop=开始使用 Loop 时是否需要帮助
read_about_loop_first=开始使用前是否阅读过在线帖子/讨论
loop_day_use=是否曾在白天使用 Loop
hard_use_during_day=白天使用 Loop 的困难程度
well_work_during_day=对白天运行效果的评价；1很好至5不好
look_app_on_phone=查看手机 Loop App 的频率
tune_radio_frequency_day=调整无线电频率的频率
hard_carb_entry=使用碳水录入功能的困难程度
enter_carbs_dont_eat=是否曾录入碳水但没有实际吃碳水；提示食物记录不一定是真实进食
carbs_low_bg=处理低血糖所吃碳水是否录入：1总是，2从不，3有时
manual_bolus_day=是否使用过 Loop 手动 Bolus 功能
workout_glucose_target=是否使用过运动血糖目标功能
workout_glucose_target_hard=使用运动血糖目标功能的困难程度
loop_night_use=是否曾在夜间使用 Loop
well_work_during_night=对夜间运行效果的评价；1很好至5不好
manual_bolus_night=是否在夜间手动给予 Bolus
loop_connect_issue_frequency=Loop 绿圈变灰/红、未成功闭环的频率
manual_connection_fix=Loop 未成功运行时是否手动修复连接
manual_fix_wait_time=通常等待多久才开始手动修复连接
trouble_getting_supplies=获取胰岛素、泵或 CGM 耗材是否有困难''')
CN={k.strip():v for k,v in CN.items()}
# Checkbox columns are separate answers, not counts.
def checkboxes(prefix,label,options,start=1):
 for i,opt in enumerate(options.split('|'),start):CN[f'{prefix}___{i}']=f'{label}：{opt}（独立多选项）'
checkboxes('insulin_injection_reason','额外注射原因','允许取下泵|减少低血糖/DKA（原题选项，不是疗效认定）|胰岛素用量较高（原文措辞较简略）|运动/活动|泵故障|泵耗材用完|其他')
checkboxes('afrezza_reason','额外吸入胰岛素原因','餐时替代餐时Bolus|餐时在Bolus之外追加|纠正高血糖时替代Bolus|纠正高血糖时在Bolus之外追加|其他')
checkboxes('aid_use_type','曾用AID','Medtronic 670G|OpenAPS|AndroidAPS|其他')
checkboxes('why_start_loop','开始Loop的原因','减少低血糖次数/持续时间|改善睡眠|改善血糖/HbA1c|更多控制胰岛素输注|界面更易用|参与开源运动|其他')
checkboxes('additional_medications','额外药物','Symlin|二甲双胍|GLP-1类似物|DPP-4抑制剂|SGLT-2抑制剂|草药/补充剂|其他|无')
checkboxes('who_share_cgm','CGM分享对象','不分享|伴侣|父母/监护人|其他家人|朋友|其他',0)
checkboxes('how_share_cgm','CGM分享方式','不分享|Nightscout|Dexcom Share|其他',0)
checkboxes('who_helped_you_start_loop','开始Loop时的帮助者','已有Loop用户|协助管理糖尿病的人|糖尿病医生/护士|糖尿病教育者|更懂计算机/技术的人')
checkboxes('what_carb_features','所用碳水录入功能','食物类型图标|手动录入其他食物|手动修改吸收时间')
checkboxes('workout_glucose_target_what','运动目标使用情境','运动/身体活动|提高血糖目标以预防低血糖|饮酒且担心低血糖')
checkboxes('how_manually_fix','手动修复方式','启动Dexcom App|用电源键软重启iPhone|组合键强制重启iPhone|在Loop中断开并重连RileyLink|关闭再开启RileyLink硬件|更换RileyLink电池|完全退出Loop App|重装相同/新版Loop|重装相同/新版RileyLink软件|关闭再开启iPhone蓝牙|其他')
checkboxes('why_stop_loop','停止使用原因','不喜欢|尝试其他方式|太贵|耗材难获取|太复杂/找不到正确使用资源|未看到本人/孩子血糖改善|不能正常工作|版本不是最新|太耗时|帮助未达预期|其他|用户社区体验不好|担心向医生透露使用情况|不想随身携带设备|自报本人/孩子已无糖尿病（不构成医学确认）|参加其他临床试验|怀孕|等待兼容Omnipod')
AID=dict(x.split('=',1) for x in '''more_hopeful=对未来更有希望
worry_less=减少对糖尿病的担忧
reduce_fam_concern=减少家人对糖尿病的担忧
easy_todo_activities=更容易做想做的事
decrease_lows=减少低血糖
decrease_highs=减少高血糖
target_range=更多时间处于目标血糖范围
improve_a1c=让HbA1c达到目标
eat_anytime=更容易在想吃时进食
exercise_anytime=更容易在想运动时运动
mnge_work_school=更容易在工作/学校管理糖尿病
mnge_social_life=更容易在社交时管理糖尿病
mnge_sick_days=更容易在生病时管理糖尿病
sleep_better=睡得更好
lows_nights=减少夜间低血糖
qual_life=改善生活质量
fam_qual_life=改善家人生活质量
mnge_drive_travel=更容易在驾车/旅行时管理糖尿病
mnge_travel=更容易在旅行时管理糖尿病
mnge_sex_life=更容易在性生活方面管理糖尿病
mnge_drink_alcohol=若选择饮酒，更容易管理糖尿病
help_pregnancy=若怀孕，更容易管理糖尿病
longterm_comp=降低长期并发症风险'''.splitlines())
FEAR='''not_rec_real_low=未发现自己低血糖
not_have_food_fruit=身边没有食物、水果或果汁
pass_out_pub=在公共场合昏倒
embarrass_soc_sit=在社交中使自己或朋友尴尬
have_react_alne=独处时发生低血糖反应
appear_stupid_drunk=看起来迟钝或像喝醉
lose_cntrl=失去控制
no_one_arnd_help=低血糖反应时身边无人帮助
have_react_driving=驾车时发生低血糖反应
mistake_have_accdnt=犯错或发生事故
bad_eval_criticize=得到差评或受到批评
diff_think_clear=需要照顾他人时难以清晰思考
lightheaded_dizzy=头昏/眩晕
not_recgnze_low=未发现自己低血糖
not_have_food=低血糖时身边没有食物、水果或果汁
feel_dizzy=因低血糖在公共场合眩晕或昏倒
low_asleep=睡眠时低血糖
embrss_bc_low=因低血糖使自己尴尬
low_by_myself=独处时低血糖
look_stupid=在他人面前显得迟钝或笨拙
lose_ctrl=因低血糖失控
no_one_arnd=低血糖时无人帮助
mistake_school=在学校犯错或发生事故
trble_school=因低血糖期间发生的事在学校遇到麻烦
have_seizure=发生惊厥
lng_term_comp=因低血糖产生长期并发症（受访者担忧内容）
dizzy_woozy=低血糖时头昏
have_low_bld_sug=发生低血糖
child_not_recog_low=孩子未发现自己低血糖
child_not_have_food=孩子身边没有食物、水果或果汁
child_dizzy_passingout=孩子在公共场合眩晕或昏倒
child_low_while_sleep=孩子睡眠时低血糖
child_embarrass_self_others=孩子在社交中使自己或亲友尴尬
child_have_low_alone=孩子独处时低血糖
child_appear_stupid_clumsy=孩子显得迟钝或笨拙
child_lose_control_behavior=孩子因低血糖行为失控
no_one_help_child_during_low=孩子低血糖时无人帮助
child_make_mistakes=孩子在学校犯错或发生事故
child_getting_a_bad_eval=孩子因低血糖期间发生的事在学校获得差评
child_having_seizure=孩子发生惊厥或抽搐
child_longterm_complications=孩子因频繁低血糖出现长期并发症（受访者担忧内容）
child_feel_faint=孩子头昏或晕厥
child_having_low=孩子发生低血糖'''
for x in FEAR.splitlines():
 k,v=x.split('=',1);CN[k]='担忧“'+v+'”的频率；不是事件实际发生次数'
# Keep raw spelling; use official section labels to separate baseline/follow-up and scales.
GROUP={
'LOOP Baseline Survey':'基线一般资料',
'LOOP Additional Baseline Survey (For Current Loop Users)':'已有Loop用户的基线补充',
'LOOP Follow-Up Survey':'随访一般资料',
'LOOP Additional Follow-Up Survey (Current Users Only)':'已有Loop用户的随访补充',
}
D=[]
for t in A[1:]:
 dic={};group='身份与阶段'
 for r in t[1:]:
  if len(r)==1:group=r[0];continue
  if len(r)<2:continue
  dic[r[0].lower()]={'official_field':r[0],'english':r[1],'min':r[2] if len(r)>2 else '', 'max':r[3] if len(r)>3 else '', 'codes':r[4] if len(r)>4 else '', 'group':group}
 D.append(dic)
HEAD={}
for f in sorted((RAW/'Data Tables').glob('*.txt')):
 with f.open('rb') as h:HEAD[f.name]=h.readline().decode('utf-8-sig').strip('\r\n').split('|')
for k in HEAD['Surveys.txt']:
 if '_aid_' in k:
  who,suffix=k.split('_aid_',1);target={'adult':'成人本人认为AID能','child':'青少年本人认为AID能','parent':'家长认为AID能帮助孩子'}[who]
  if who=='parent' and suffix in ('sleep_better','qual_life'):target='家长认为AID能让家长自己'
  CN[k]=target+AID[suffix]+'；属于态度/期望'
 if k not in CN:
  base=k.replace('_f___','___').replace('cgmf___','cgm___')
  if base.endswith('_f'):base=base[:-2]
  elif base.endswith('f'):base=base[:-1]
  if base in CN:CN[k]='随访问项：'+CN[base]
assert all(k in CN for k in HEAD['Surveys.txt']),[k for k in HEAD['Surveys.txt'] if k not in CN]
INDEX={'LOOPContactInteraction':1,'LOOPDeviceBasal':2,'LOOPDeviceBGM':3,'LOOPDeviceBolus':4,'LOOPDeviceCGM':5,'LOOPDeviceExercise':6,'LOOPDeviceFood':7,'LOOPDeviceIssueRpt':8,'LOOPDeviceUploads':9,'LOOPDeviceWizard':10,'LOOPPtFinalStatus':11,'PtRoster':12,'SampleResults':13,'gluIndices':14,'Surveys':15,'adverseEvents':16,'deviceDiscontSurvey':17,'deviceIssues':18}
INFO={
'LOOPContactInteraction':('研究联系记录','研究人员与参与者的一次联系记录，不是完整病历或对话。'),
'LOOPDeviceBasal':('基础胰岛素输注事件','一次基础率事件，含实际生效时长、基础率和被覆盖事件的信息；3个文件是同结构分卷。'),
'LOOPDeviceBGM':('血糖仪点测记录','一次血糖仪测量记录，不能当作连续CGM。'),
'LOOPDeviceBolus':('大剂量胰岛素输注事件','一次泵Bolus记录，普通部分和延长部分、预期量和记录执行量需分别理解。'),
'LOOPDeviceCGM':('连续血糖与校准记录','一条设备血糖/校准记录；6个文件是同结构分卷，并非6个患者或训练测试划分。'),
'LOOPDeviceExercise':('运动事件','一次上报运动，含距离、时长、能量等；不是所有患者都有完整的连续活动监测。'),
'LOOPDeviceFood':('碳水录入事件','一次净碳水录入；缺少逐餐食物图片、完整营养成分及摄入真实性验证。'),
'LOOPDeviceIssueRpt':('Loop配置与问题报告','一次报告的配置快照/详细内容；不是每次决策都记录的完整状态。'),
'LOOPDeviceUploads':('设备上传元数据','一条设备数据上传记录，是多种事件表的父记录。'),
'LOOPDeviceWizard':('Bolus计算器记录','一次计算器输入及建议，包含部分IOB和参数；不是实际给药执行记录。'),
'LOOPPtFinalStatus':('研究退出/最终状态记录','一次研究退出及末次问卷意愿等行政记录。'),
'PtRoster':('参与者名册','参与者入组、队列、年龄和状态；理解全数据集的患者入口。'),
'SampleResults':('实验室检验结果','一项样本检验结果，如HbA1c；采样、分析日期和单位单列。'),
'gluIndices':('阶段血糖与胰岛素统计','一个患者一个研究阶段的汇总指标；是官方附带的既有派生结果，不是本轮新处理的产物。'),
'Surveys':('多问卷答案宽表','某参与者某阶段的多问卷答案汇集，330字段；不同年龄/队列/阶段适用的问题不同。'),
'adverseEvents':('不良事件简表','一次上报不良事件的类型和日期；不是全量自动检测的低血糖标签。'),
'deviceDiscontSurvey':('停止使用Loop原因','参与者勾选的停用原因，多选字段分列；不等于所有人都退出研究。'),
'deviceIssues':('设备问题简表','一次设备问题的日期、问题类型和设备；区别于设备详细配置报告。')}
def logical(name):return re.sub(r'\d+$','',Path(name).stem)
def esc(s):return str(s).replace('|','&#124;').replace('\n','<br>').replace('\r','')
def clean(s):return '; '.join(x.strip() for x in s.splitlines() if x.strip())
SCHEMA=[];MISSING=[];EXTRA=[]
for name,cols in HEAD.items():
 key=logical(name);dic=D[INDEX[key]-1];fields=[]
 for k in cols:
  assert k in CN,(name,k)
  d=dic.get(k.lower(),{}).copy()
  if not d:MISSING.append([name,k])
  d.update(field=k,chinese=CN[k]);fields.append(d)
 SCHEMA.append({'file':name,'logical_table':key,'columns':len(cols),'fields':fields})
 extras=[r['official_field'] for k,r in dic.items() if k not in {c.lower() for c in cols}]
 if extras:EXTRA.append([name,extras])
(P/'实际字段清单.json').write_text(json.dumps({'tables':SCHEMA,'dictionary_missing_fields':MISSING,'dictionary_only_fields':EXTRA},ensure_ascii=False,indent=2))
# Main non-survey dictionary: no field omitted, shared physical partitions explained once.
lines=['# Loop 数据表逐字段说明','', '2026-09-14；依据实际TXT表头与官方 DataGlossary.rtf。本文件解释除 Surveys 外的17种逻辑表；[Surveys的330字段单列](03_Surveys全部330字段说明.md)。原始列名、大小写、拼写均保留。','',
'中文为释义，英文列保留官方原文以便核对。字典没有写出的单位、枚举和关联不自动补全。表中“未列”不表示实际值为空；缺失值也不代表0。Min/Max是字典列出的范围，不是本次扫描得出的实际最值。','']
seen=set()
for s in SCHEMA:
 key=s['logical_table']
 if key in seen or key=='Surveys':continue
 seen.add(key);names=[x['file'] for x in SCHEMA if x['logical_table']==key]
 assert all(HEAD[n]==HEAD[names[0]] for n in names)
 lines += [f'## {key}：{INFO[key][0]}','',INFO[key][1],'','文件：'+ '、'.join(f'`{n}`' for n in names)+f'；每个文件 {s["columns"]} 列。','']
 if key=='LOOPDeviceBasal':lines+=['`Duration` 和 `ExpectedDuration` 的官方单位均为毫秒；`Rate` 为 U/h。`Supr*` 描述被当前事件覆盖的另一基础事件，不能相加。本轮不重建区间、不计算区间剂量。','']
 if key=='LOOPDeviceBolus':lines+=['`Duration`/`ExpectedDuration`：Tidepool来源为毫秒，Diasend来源为分钟。需结合上传来源；不能因为同名就与另一来源直接拼接。普通量和延长量按原列保留。','']
 if key=='LOOPDeviceIssueRpt':lines+=['历史全量审计保留了4条复杂记录列宽异常；文件原样保留。字典未定义嵌套设置的完整结构及所有单位。不能简单丢坏行后声称全部读取。','']
 if key=='gluIndices':lines+=['这里的均值/标准差使用 mg/dL，原始CGM的Tidepool读数使用 mmol/L；这是两个不同层次的文件。此表已汇总未来整个阶段，不是当时决策可直接看到的实时输入。','']
 lines+=['| 原始字段 | 中文含义 / 单位 / 注意事项 | 官方英文释义 | 官方枚举或字典范围 |','|---|---|---|---|']
 for f in s['fields']:
  v=clean(f.get('codes','')) or '未列'
  if f.get('min') or f.get('max'):v+=f"；Min={f.get('min') or '未列'}；Max={f.get('max') or '未列'}"
  lines.append('| '+ ' | '.join(esc(x) for x in [f['field'],f['chinese'],f.get('english') or '官方未提供该字段释义',v])+' |')
 lines+=['']
lines+=['## 字典与实际表头的差异','', '- 多数设备/名册表实际有 `PtID`，字典未单列该字段；本说明以实际表头补列参与者编号。','- `SubjectID`、`subjectID` 与字典中的 `SubjectId` 大小写不统一；`adverseEvents`等表的字段大小写也不同，原始文件不改名。','- `BolusRecID` 的官方释义为空，不能把从字段名推断的关联当成已验证外键。','- `Surveys`字典中的 `typical_cgm_locationf` 在本次官方包的实际表头中不存在；不得虚构为第331列。','']
(DOC/'02_数据表字段字典.md').write_text('\n'.join(lines))
# Survey dictionary with full codes once per unique codebook, and anchors from every field.
s=next(x for x in SCHEMA if x['logical_table']=='Surveys');codes={};codelist=[]
lines=['# Surveys.txt 全部330字段说明','', '[返回文件总览](01_数据集与全部文件说明.md) · [其余表字段](02_数据表字段字典.md)','',
'这是一张问卷答案宽表，不是连续监测表。每行汇集参与者在某研究阶段的问卷答案，用 SubjectID 与 Period 理解。按年龄、队列和随访阶段分发的问卷不同，大量空格可能是未适用/跳题/未答，不能统一当作否或0。','',
'下文保持实际330列表头顺序，每列给出中文释义、官方英文题干和编码引用。_f 或末尾 f 为相应随访问项，不能据后缀推断它已在基线时可知；___数字是多选题选项编号，列值是勾选状态。','',
'常见编码：Yes/No=是/否，Checked/Unchecked=已选/未选，N/A=不适用；Strongly disagree→Strongly agree=非常不同意→非常同意；Never/Rarely/Sometimes/Often/Almost always=从不/很少/有时/经常/几乎总是。完整值与各题具体方向见文末编码表，不能把所有问卷当成同一种0–5分。','',
'关键例外：pregnant 的2表示否；carbs_low_bg 的1/2/3分别表示总是/从不/有时；rate_overall_sleep_quality 的字典编码为1很差到4很好，不能套用其他PSQI版本的0–3。weight 是磅；height_feet/height_inches 是英尺和英寸。','',
'低血糖恐惧题的分数表示“担忧频率”，不是低血糖次数。INSPIRE反映对AID的态度/期望，不是系统已实现的医学效果。成人/青少年恐惧量表PDF仅是版权提示，但字段和题干仍见官方字典；家长恐惧题的字典未列编码，本说明明确依据 Parent Low Blood Sugar Survey.pdf 第2页补充0–4频率编码。公开列仅含该家长量表的Worry部分，不含第1页全部行为题。','']
last=None;number=0
for f in s['fields']:
 group=f['group']
 if group!=last:
  lines += ['',f'## {GROUP.get(group,group)}','', '| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |','|---|---|---|---|---|'];last=group
 number+=1;c=clean(f.get('codes',''));note=''
 if f['field'] in [x.split('=',1)[0] for x in FEAR.splitlines()][28:]:
  assert not c
  c='0 = Never; 1 = Rarely; 2 = Sometimes; 3 = Often; 4 = Almost always';note='（来源：家长问卷PDF第2页，字典未列）'
 if c:
  if c not in codes:codes[c]=f'C{len(codes)+1:02d}';codelist.append((codes[c],c))
  ref=f'[{codes[c]}](#{codes[c].lower()})'+note
 else:ref='字典未列枚举；按题意读取'
 if f.get('min') or f.get('max'):ref+=f"；Min={f.get('min') or '未列'}，Max={f.get('max') or '未列'}"
 lines.append('| '+' | '.join(esc(x) for x in [number,f['field'],f['chinese'],f.get('english',''),ref])+' |')
lines+=['','## 全部答案编码','', '以下为官方字典的完整可选值，去除排版空行，不改代码。家长恐惧题的补充来源已逐行标注。无枚举不等于无约束，也不等于不存在缺失编码；本轮不转换任何原始值。','']
for label,c in codelist:lines += [f'### {label}','',c,'']
lines += ['## 版本差异','', '`typical_cgm_locationf` 在官方字典中有定义，但本包 Surveys.txt 实际没有这一列。本说明不把它计入330列，也不从其他列生成它。`finacial_risk` 是原始拼写，保持不改。','']
(DOC/'03_Surveys全部330字段说明.md').write_text('\n'.join(lines))
print(json.dumps({'physical_tables':len(HEAD),'logical_tables':len(INDEX),'survey_columns':len(HEAD['Surveys.txt']),'physical_field_occurrences':sum(map(len,HEAD.values())),'logical_field_occurrences':sum(len(HEAD[next(n for n in HEAD if logical(n)==k)]) for k in INDEX),'dictionary_only_fields':EXTRA,'cn_coverage':'complete'},ensure_ascii=False))
PDF_INFO={
'Loop Protocol v4.0 18 Jul 2019.pdf':'研究方案第4版（2019-07-18）：研究设计、队列、随访安排、采集内容与安全事件定义；不是数据表。',
'Device Discontinuation - Loop.pdf':'停用Loop原因问卷；答案在 deviceDiscontSurvey.txt 的18个选项列中。',
'Diabetes Technology Attitudes.pdf':'糖尿病技术态度5题；对应 Surveys 中 tech_ 开头的5列。',
'Hypoglycemia Confidence Scale.pdf':'低血糖应对信心量表；对应 when_exrcse 至 confdt_spouse 共9列。',
'Hypoglycemia Fear Survey Adult.pdf':'本包仅含版权提示和参考信息，未提供量表正文；成人恐惧题的公开答案和题干分别在 Surveys 与 DataGlossary 中。',
'Hypoglycemia Fear Survey Child.pdf':'本包仅含版权提示和参考信息，未提供量表正文；青少年恐惧题的公开答案和题干分别在 Surveys 与 DataGlossary 中。',
'INSPIRE Questionnaire for Adults.pdf':'成人对AID的态度/期望问卷，对应 adult_aid_ 开头的列。',
'INSPIRE Questionnaire for Parents.pdf':'家长对孩子使用AID的态度/期望问卷，对应 parent_aid_ 开头的列；部分问题询问家长自己的睡眠/生活质量。',
'INSPIRE Questionnaire for Youth.pdf':'青少年对AID的态度/期望问卷，对应 child_aid_ 开头的列。',
'Loop Additional Baseline Form for Current Loop Users.pdf':'已有Loop用户的额外基线表：版本、分支、使用时段、CGM分享、Apple Watch和医护互动等；公开字段并非包含表单全部题目。',
'Loop Adverse Event and Device Issue Form.pdf':'不良事件和设备问题报告表；公开简表主要为 adverseEvents.txt 与 deviceIssues.txt，不包含本表所有细节。',
'Loop Follow-Up Form for Current Loop Users.pdf':'已有Loop用户的额外随访表：版本、使用情况、分享与医护互动；对应Surveys额外随访组。',
'Loop Follow-Up General Data Collection.pdf':'一般随访资料：孕况、治疗、CGM、额外药物、低血糖感知等；对应Surveys随访组。',
'Loop baseline general data collection.pdf':'一般基线资料：人口、身高体重、病程、HbA1c、泵/CGM、胰岛素/其他药物、事件史及运动习惯；对应Surveys基线组。',
'LoopHoles Survey.pdf':'Loop可用性/行为体验：信任、使用困难、碳水录入、手动Bolus、连接中断、修复与耗材获取；对应Surveys的Loopholes组。',
'Parent Low Blood Sugar Survey.pdf':'家长低血糖调查。公开Surveys列对应第2页Worry部分15题；第1页行为题没有全部发布为对应列。第2页给出0–4担忧频率编码。',
'Pittsburgh Sleep Quality Index.pdf':'过去一个月睡眠问卷；对应Surveys睡眠相关17列，是问卷，不是手表睡眠时序。字典编码以本包为准。',
'Risk Taking Questions.pdf':'总体、财务、职业、健康四类冒险意愿；对应Surveys的4个risk列。',
'T1-DDS management distress items.pdf':'T1D管理困扰的4个条目；对应Surveys的4个管理困扰字段，不是全套T1-DDS。',
'Technology Use for Problem Solving.pdf':'借助技术解决血糖问题的频率，9题；对应Surveys的TUPS字段。',
}
manifest=json.loads((P/'原始文件校验.json').read_text());pdfmeta={x['file']:x for x in json.loads((P/'PDF文档清单.json').read_text())}
lines=['# Loop 数据集：内容、全部文件与阅读方法','',
'版本：官方 `Loop study public dataset 2023-01-31.zip`；说明日期：2026-09-14。本文依据本地官方ReadMe、字典、研究方案、问卷和实际表头。历史规模/质量数字明确标注为2026-09-12已有审计，本轮未重新清洗或构建训练数据。','',
'## 1. 先回答：那些TXT是不是数据？','',
'**是。`原始数据/Data Tables/` 中25个TXT就是官方原始数据表，第一行是列名，列之间用竖线 `|` 分隔。** 文件扩展名是TXT不表示它是说明文字。CGM有6个分卷，基础输注有3个分卷，分卷内结构一致；其余每种表一个文件。合计18种逻辑表。','',
'不要逐个把几GB的TXT当普通文档打开，也不要直接用Excel载入全表（容量不足）。目前只需对照本说明阅读字段；本轮已只读核对表头，没有导出新的CSV、插值表或对齐表。','',
'官方包另外包含2份RTF、1份研究方案PDF、19份问卷PDF。因此47个官方文件=25 TXT+2 RTF+20 PDF。macOS生成的 `.DS_Store` 是Finder元数据，保留但不计入官方47文件。','',
'## 2. 它记录的是什么','',
'这是Loop Observational Study的真实世界观察性研究公开数据。参与者为使用或准备开始使用Loop自动胰岛素输注系统的T1D成人与青少年。它同时记录设备事件、参与者资料、问卷与研究随访。A队列是新用户，B队列是既有用户；不是随机分配各种新剂量的试验库。','',
'理解数据可以按五层阅读：','',
'| 层次 | 代表文件 | 能看到什么 |','|---|---|---|',
'| 人是谁、何时入组 | PtRoster、Surveys | 去标识化编号、年龄、队列、病程、体重、治疗和生活习惯 |',
'| 血糖发生了什么 | CGM1–6、BGM、SampleResults | 连续血糖/校准、点测血糖、实验室指标 |',
'| 记录了哪些干预和活动 | Basal1–3、Bolus、Food、Exercise | 基础输注、Bolus、录入碳水、运动事件 |',
'| 设备当时知道/设置了什么 | Wizard、IssueRpt、Uploads | 计算器建议/IOB、稀疏配置快照、设备与上传来源 |',
'| 整个研究阶段的结局/体验 | gluIndices、Surveys、adverseEvents、deviceIssues、停用/退出/联系表 | 官方汇总、主观体验、不良事件和行政记录 |','',
'它不是包含完整电子病历、所有进食、所有额外注射、每分钟心率和睡眠的统一多模态矩阵。运动事件和问卷确实是额外信息，但不能等同于每个患者都有同步完整的可穿戴数据。','',
'## 3. 规模：文件完整与临床记录完整要分开','',
'本轮文件校验：原ZIP为1,626,077,053字节；47个官方解压文件合计20,660,424,864字节（约20.66GB，19.24GiB）。每个文件的大小/CRC均与ZIP一致，并新记录SHA-256。原始文件没有改写。','',
'以下规模来自[2026-09-12历史审计](../历史资料/2026-09-12审计/audit.md)，不是本轮新筛选：','',
'| 数据 | 原始记录规模 | 解释 |','|---|---:|---|',
'| PtRoster | 919人 | 入组成人548；A队列620，B队列299 |',
'| CGM六卷 | 111,118,148行 | 含58,728条Calibration；CGM-only为111,059,420行，851人 |',
'| Basal三卷 | 48,301,308行 | 845人；日志有重复、重叠与特殊状态，不能把行数当独立动作数 |',
'| Bolus | 2,722,513行 | 845人 |',
'| Food | 1,406,204行 | 837人；记录碳水不保证每餐实际摄入均完整 |',
'| Exercise | 51,728行 | 493人；覆盖远小于CGM |',
'| Wizard | 119,354行 | 215人；IOB并非全患者连续提供 |',
'| Surveys | 2,202行、330列 | 826人；Baseline826、Month3 508、Month6 701、Month12 167行 |','',
'原始包人数与论文主要分析人数口径不同，也包含入组前的上传历史。919人或此前提名的369人都不是本轮确认的可训练人数。','',
'## 4. 25个TXT逐文件说明','',
'下表逐个列出物理文件；分卷仍逐个列出，但字段说明共享对应逻辑表。文件名可直接点击。每行的含义是表的记录粒度，不表示已经检查过唯一性。','',
'| 文件 | 字节数 | 列数 | 一行表示什么 / 文件用途 |','|---|---:|---:|---|']
for name,cols in HEAD.items():
 f=next(x for x in manifest['files'] if x['file']=='Data Tables/'+name);key=logical(name);note=INFO[key][1]
 if key in ('LOOPDeviceCGM','LOOPDeviceBasal'):note=f'第{Path(name).stem[-1]}分卷。'+note
 lines.append(f'| [{name}](<../原始数据/Data Tables/{name}>) | {f["bytes"]:,} | {len(cols)} | {note} |')
lines += ['','所有字段见[数据表逐字段字典](02_数据表字段字典.md)与[Surveys全部330字段](03_Surveys全部330字段说明.md)。两份字典合计覆盖18种逻辑表的596个字段位置；同名字段在不同表仍逐表解释。','',
'## 5. 2份RTF与20份PDF逐文件说明','',
'这些是数据发布自带的说明/研究材料，保留在原目录。`Participant Surveys`中的PDF是空白问卷模板/说明，不是每个患者的PDF病例；患者答案在TXT表中。公开数据为去标识化子集，不能因为问卷中有某题就认定TXT已发布对应字段。','',
'| 官方文件 | 字节数 | 页数 | 内容及与数据的关系 |','|---|---:|---:|---|']
for f in manifest['files']:
 name=f['file']
 if name.startswith('Data Tables/'):continue
 if name.endswith('.pdf'):desc=PDF_INFO[Path(name).name];pages=pdfmeta[name]['pages']
 else:
  desc={'DataGlossary.rtf':'官方数据字典：逐表字段、英文含义、部分范围和答案编码。本说明的主要来源；与实际表头存在少量差异，已逐项标记。','LOOP ReadMe.rtf':'官方发布说明：文件组织、去标识化日期、研究论文与数据归因要求。'}[name];pages='—'
 lines.append(f'| [{name}](<../原始数据/{name}>) | {f["bytes"]:,} | {pages} | {desc} |')
lines += ['','## 6. 文件之间怎么对应','',
'首先用患者编号理解同一人的记录：设备、名册和检验表常用 `PtID`；问卷/统计/问题简表用 `SubjectID` 或 `subjectID`。这些是公开去标识化参与者编号，实际大小写必须保留；不能拿行号当患者编号。','',
'设备事件中的 `ParentLOOPDeviceUploadsID` 明确指向 `LOOPDeviceUploads.RecID`，可追溯上传来源。`RecID` 本身是本表记录编号，不是患者身份。设备事件通常多条对应一条上传，而一人有多次上传；不是一对一横向拼表。','',
'`Wizard.BolusRecID` 从命名看可能链接Bolus，但官方解释为空，本轮未证明外键完整性。`Surveys`和`gluIndices`还有阶段 `Period`；阶段标签不是精确测量时刻。不能把随访问卷和阶段结局直接复制到这一患者的所有历史时刻。','',
'```text\n参与者名册 PtRoster\n  ├─ 同一患者的设备上传 LOOPDeviceUploads\n  │    └─ ParentLOOPDeviceUploadsID → 上传表RecID\n  │         ├─ CGM / BGM\n  │         ├─ Basal / Bolus / Food / Exercise\n  │         └─ Wizard / IssueRpt\n  ├─ 问卷 Surveys（再区分Period）\n  ├─ 检验 SampleResults（再区分Visit和采样日期）\n  ├─ 阶段统计 gluIndices（再区分Period）\n  └─ 不良事件 / 设备问题 / 停用 / 退出 / 联系记录\n```','',
'这张图描述官方字段提供的关联线索，不代表本轮已执行全量关联或验证所有患者/外键的一致性。','',
'## 7. 时间、单位和空值应怎样读','',
'官方ReadMe说明：参与者编号和日期已去标识化，每个参与者日期随机移动最多365天，同一人的相对时间关系保留。因此可理解个人的相对先后，但不能把跨患者同一天解释成真实共同季节/节假日/天气暴露。','',
'事件时间、设备本地时间、上传时间、报告生成时间、研究阶段是不同概念。`UTCDtTm`的官方措辞涉及时区调整，`TmZnOffset`许多记录缺失；不能为了“看起来统一”重复套用时差。上传时间也不等于事件发生时间，更不能据上传顺序判定实际治疗顺序。','',
'| 量 | 对应原字段 | 本包解释 |','|---|---|---|',
'| 连续血糖 | CGMVal + Units | Tidepool原始CGM为mmol/L |',
'| 阶段平均血糖 | gluIndices.gluMean | mg/dL |',
'| 基础率 | Basal.Rate | U/h |',
'| 基础事件时长 | Basal.Duration | 毫秒 |',
'| 普通/延长Bolus量 | Normal / Extended | U；Expected*另指预期量 |',
'| Bolus时长 | Bolus.Duration | Tidepool毫秒，Diasend分钟；依来源 |',
'| 食物净碳水 | CarbsNet + CarbUnits | 历史审计本包为g；仍保留单位字段 |',
'| 体重 | Surveys.weight | 磅lbs |',
'| 身高 | height_feet + height_inches | 英尺部分与英寸部分 |',
'| 运动量 | Value + Units各组 | 分别读取距离、时长、能量单位 |',
'| Wizard输入血糖/碳水 | BgInput / CarbInput | 字典都写mg，含义/单位存在疑点，不能静默纠正 |','',
'空值可能是没上传、设备不提供、问卷未适用、未填写或特定事件的表达，不统一等于0。例如历史审计发现大多数Basal.Rate空值属于suspend暂停事件；这一状态须按设备数据语义解释，不能用普通数值填补规则处理。','',
'## 8. 本轮确认的数据理解边界','',
'- 文件传输完整已核验；临床事件记录连续、无冲突、进餐/用药完整并未因此成立。历史审计发现CGM重复/冲突、基础事件重叠和长缺口，原始内容均保留。',
'- `BasalProfile`是设置，`Basal.Rate`结合事件时长才是实际基础事件；`Supr*`为被覆盖事件。`Wizard`是建议/估计，`Bolus`是输注日志。泵输注日志也不直接测量人体实际吸收。',
'- `Food`仅证明系统有碳水录入。问卷专门询问“录入但没吃”，并询问低血糖摄入是否记录，因此不能把这一表直接称为完整真实饮食。',
'- IssueRpt存在4条复杂记录宽度异常及非UTF8内容（历史审计）；本轮没有跳过、改码或修复原文件。',
'- 成人/青少年恐惧量表PDF只有版权提示；家长量表完整PDF和公开字段覆盖范围不同；`typical_cgm_locationf`仅在字典出现，实际Surveys没有这一列。',
'- 本轮产物是目录整理和字段解释，没有新患者筛选、训练/测试划分、插值、时间对齐、合并数据表、模型训练或Harness接入。','',
'## 9. 归档与证据位置','',
'- 原始文件：[原始数据](../原始数据/)；原ZIP：[官方压缩包](../官方压缩包/)。',
'- 旧统计/脚本：[历史资料](../历史资料/00_历史资料说明.md)。这些不是另一套Loop原始数据。',
'- [原始文件SHA-256/CRC校验](核验资料/原始文件校验.json)、[迁移清单](核验资料/迁移清单.json)、[实际字段清单](核验资料/实际字段清单.json)。字段清单只包含表头和释义，无患者数据行。',
'- `datasets/`仍有跨多个数据集的共同报告/索引，不能整体搬走；其中Loop专属原包、解压数据、申请记录、下载材料与历史审计已全部迁出。旧历史日志/脚本保留当时路径以便追溯，不能不改配置直接复跑。','',
'## 10. 官方归因声明','',
'The source of the data is the Loop Study (sponsored by the Jaeb Center for Health Research and funded by the Helmsley Charitable Trust), but the analyses, content and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the study sponsor.','']
(DOC/'01_数据集与全部文件说明.md').write_text('\n'.join(lines))
(BASE/'00_从这里开始.md').write_text('''# Loop 数据集，从这里开始

**你要找的原始TXT在：[原始数据 → Data Tables](<原始数据/Data Tables/>)。**

CGM1–6是连续血糖的6个分卷，Basal1–3是基础胰岛素输注的3个分卷，Bolus是大剂量输注，Food是碳水录入。全部25个TXT使用竖线 `|` 分隔，文件内第一行是字段名。

```text
Loop数据集/
├── 原始数据/          ← 官方包解压内容，原名原内容
│   ├── Data Tables/   ← 25个TXT数据文件
│   ├── Participant Surveys/ ← 19份问卷PDF（不是患者病例）
│   ├── DataGlossary.rtf      ← 官方字段字典
│   ├── LOOP ReadMe.rtf       ← 官方使用说明
│   └── Loop Protocol v4.0 18 Jul 2019.pdf
├── 官方压缩包/        ← 原始ZIP备份
├── 说明文档/          ← 本次中文说明和文件校验凭据
└── 历史资料/          ← 之前的审计、下载和申请记录
```

| 你想知道什么 | 打开哪份说明 |
|---|---|
| 有哪些数据、每个文件是什么、怎么关联 | [01 全部文件说明](说明文档/01_数据集与全部文件说明.md) |
| CGM、胰岛素、餐食、设备等字段什么意思 | [02 数据表逐字段字典](说明文档/02_数据表字段字典.md) |
| 问卷330列每题是什么、答案数字怎么解释 | [03 Surveys全部字段](说明文档/03_Surveys全部330字段说明.md) |
| 旧的统计与申请资料是什么 | [历史资料说明](历史资料/00_历史资料说明.md) |

47个官方文件已与原ZIP逐一校验大小和CRC，并记录SHA-256；约20.66GB。80个Loop专属文件从datasets迁出，迁移时保持文件内容、inode、大小和修改时间。原始目录内的 `.DS_Store` 是已有Finder元数据，不属于官方47个文件。

本次只整理目录、读取表头和官方文档、编写说明；没有清洗、插值、时间对齐、生成训练表或训练模型。历史资料中的旧统计是先前审计结果，不能误当作本次新处理的结果。**本轮到此停止。**
''')
(BASE/'历史资料/00_历史资料说明.md').write_text('''# 历史资料：不是原始数据

这里保存迁移前已经存在的Loop专属材料。真正的数据在 [原始数据](../原始数据/)；不需要为了看数据阅读这里的脚本和日志。

| 目录 | 内容与作用 |
|---|---|
| 2026-09-12审计 | 旧审计报告、聚合统计、每患者诊断指标、脚本及日志；RTF/txt是当时的字典副本/转换件，不是另一套患者原始表 |
| 获取记录 | loop_download.json保存原始下载/校验信息；日期子目录保存官方表单页面快照和传输日志 |
| 私密申请记录 | 已有申请回执，保留本地权限限制，不属于公开患者数据 |

旧脚本、JSON、日志中的原路径按历史事实保留；部分路径随此次迁移已失效，不应直接复跑。此次仅移动它们，没有运行旧处理脚本。datasets中多来源共享的历史索引/报告继续保留，现行导航已改到Loop新位置。
''')
print('Written overview with all47 official file links and both full dictionaries.')
