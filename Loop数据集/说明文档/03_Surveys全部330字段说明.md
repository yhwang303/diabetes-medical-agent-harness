# Surveys.txt 全部330字段说明

[返回文件总览](01_数据集与全部文件说明.md) · [其余表字段](02_数据表字段字典.md)

这是一张问卷答案宽表，不是连续监测表。每行汇集参与者在某研究阶段的问卷答案，用 SubjectID 与 Period 理解。按年龄、队列和随访阶段分发的问卷不同，大量空格可能是未适用/跳题/未答，不能统一当作否或0。

下文保持实际330列表头顺序，每列给出中文释义、官方英文题干和编码引用。_f 或末尾 f 为相应随访问项，不能据后缀推断它已在基线时可知；___数字是多选题选项编号，列值是勾选状态。

常见编码：Yes/No=是/否，Checked/Unchecked=已选/未选，N/A=不适用；Strongly disagree→Strongly agree=非常不同意→非常同意；Never/Rarely/Sometimes/Often/Almost always=从不/很少/有时/经常/几乎总是。完整值与各题具体方向见文末编码表，不能把所有问卷当成同一种0–5分。

关键例外：pregnant 的2表示否；carbs_low_bg 的1/2/3分别表示总是/从不/有时；rate_overall_sleep_quality 的字典编码为1很差到4很好，不能套用其他PSQI版本的0–3。weight 是磅；height_feet/height_inches 是英尺和英寸。

低血糖恐惧题的分数表示“担忧频率”，不是低血糖次数。INSPIRE反映对AID的态度/期望，不是系统已实现的医学效果。成人/青少年恐惧量表PDF仅是版权提示，但字段和题干仍见官方字典；家长恐惧题的字典未列编码，本说明明确依据 Parent Low Blood Sugar Survey.pdf 第2页补充0–4频率编码。公开列仅含该家长量表的Worry部分，不含第1页全部行为题。


## 身份与阶段

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 1 | SubjectID | 去标识化参与者编号；不同表实际拼写大小写不同 | User ID | 字典未列枚举；按题意读取 |
| 2 | Period | 研究阶段/统计区间标签；不是逐条事件时间戳 | Study Period | 字典未列枚举；按题意读取 |
| 3 | PtCohort | 研究队列：A 新使用 Loop，B 已在使用 Loop | Participant Cohort | 字典未列枚举；按题意读取 |

## 基线一般资料

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 4 | ageAtBaseline | 基线入组年龄 | Age at study enrollment | 字典未列枚举；按题意读取 |
| 5 | gender | 参与者性别（官方题目 gender） | Participant gender | [C01](#c01) |
| 6 | ethnicity | 是否为西班牙裔/拉丁裔 | Do you consider yourself to be Hispanic/Latino or not Hispanic/Latino? | [C02](#c02) |
| 7 | race | 自报种族类别 | Which of the following racial designations best describes you? | [C03](#c03) |
| 8 | race_multiple | 多种族的自由文字补充 | Please list each race: | 字典未列枚举；按题意读取 |
| 9 | pregnant | 当前是否怀孕；特别注意 2=否，不是0 | Are you currently pregnant? | [C04](#c04) |
| 10 | height_feet | 身高的英尺部分；不是总身高厘米值 | Height (feet) | 字典未列枚举；按题意读取 |
| 11 | height_inches | 身高的英寸部分；与英尺部分配套 | Height (inches) | 字典未列枚举；按题意读取 |
| 12 | weight | 自报体重，磅（lbs），不是 kg | What is your weight (lbs)? | 字典未列枚举；按题意读取 |
| 13 | age_diabetes_developed | 发生糖尿病时的年龄 | How old were you when you developed diabetes? | 字典未列枚举；按题意读取 |
| 14 | duration | T1D 病程；字典未单列单位，不能与设备 Duration 混同 | Duration of type 1 diabetes | 字典未列枚举；按题意读取 |
| 15 | hba1c_when_measured | 距最近一次 HbA1c 测量的时长类别 | Most recent HbA1c When measured: | [C05](#c05) |
| 16 | hba1c_level | 自报最近 HbA1c 水平；不同于 SampleResults 的实验室记录 | Most recent HbA1c HbA1c level: | 字典未列枚举；按题意读取 |
| 17 | low_carb_diet | 是否遵循低碳水饮食 | Do you follow a low-carbohydrate diet? | [C06](#c06) |
| 18 | pump_use | 使用胰岛素泵的经历/当前使用状态 | What is your experience using an insulin pump? | [C07](#c07) |
| 19 | age_first_used_pump | 首次使用胰岛素泵的年龄 | At about what age did you  first use an insulin pump? | 字典未列枚举；按题意读取 |
| 20 | pump_use_length | 已经使用胰岛素泵的时长类别 | How long have you been using an insulin pump? | [C08](#c08) |
| 21 | insulin_type | 泵内使用的胰岛素制剂类别 | What type of insulin are you using in your pump? | [C09](#c09) |
| 22 | insertion_length | 通常多少天更换输注套件 | On average, how many days do you usually leave your pump insertion set in before you change it? | 字典未列枚举；按题意读取 |
| 23 | insulin_injection | 是否在用泵之外还规律注射胰岛素；不是逐次注射剂量日志 | Do you regularly use insulin injections in addition to your insulin pump therapy? | [C06](#c06) |
| 24 | insulin_injection_reason___1 | 额外注射原因：允许取下泵（独立多选项） | Allow removal of pump | [C10](#c10) |
| 25 | insulin_injection_reason___2 | 额外注射原因：减少低血糖/DKA（原题选项，不是疗效认定）（独立多选项） | Reduce hypo/DKA | [C10](#c10) |
| 26 | insulin_injection_reason___3 | 额外注射原因：胰岛素用量较高（原文措辞较简略）（独立多选项） | Insulin dose high enough | [C10](#c10) |
| 27 | insulin_injection_reason___4 | 额外注射原因：运动/活动（独立多选项） | Sports/Activities | [C10](#c10) |
| 28 | insulin_injection_reason___5 | 额外注射原因：泵故障（独立多选项） | Pump not working | [C10](#c10) |
| 29 | insulin_injection_reason___6 | 额外注射原因：泵耗材用完（独立多选项） | Out of pump supplies | [C10](#c10) |
| 30 | insulin_injection_reason___7 | 额外注射原因：其他（独立多选项） | Other | [C10](#c10) |
| 31 | afrezza | 是否在用泵之外还规律使用 Afrezza 吸入胰岛素 | Do you regularly use Afrezza (inhaled insulin) in addition to your insulin pump therapy? | [C06](#c06) |
| 32 | afrezza_reason___1 | 额外吸入胰岛素原因：餐时替代餐时Bolus（独立多选项） | Take at mealtime instead of mealtime bolus | [C10](#c10) |
| 33 | afrezza_reason___2 | 额外吸入胰岛素原因：餐时在Bolus之外追加（独立多选项） | Take at mealtime in addition to mealtime bolus | [C10](#c10) |
| 34 | afrezza_reason___3 | 额外吸入胰岛素原因：纠正高血糖时替代Bolus（独立多选项） | Take to correct highs instead of bolus | [C10](#c10) |
| 35 | afrezza_reason___4 | 额外吸入胰岛素原因：纠正高血糖时在Bolus之外追加（独立多选项） | Take to correct highs in addition to bolus | [C10](#c10) |
| 36 | afrezza_reason___5 | 额外吸入胰岛素原因：其他（独立多选项） | Other | [C10](#c10) |
| 37 | how_calculate_bolus | 餐时通常如何决定 Bolus 量 | At the time of a meal, how do you usually decide the amount of insulin to take (assuming your blood sugar it not low)? | [C11](#c11) |
| 38 | cgm_experience | CGM 使用经历/当前使用状态 | What is your experience with using a continuous glucose monitor (CGM)? | [C12](#c12) |
| 39 | age_started_cgm | 首次使用 CGM 的年龄 | At about what age did you first start using a CGM? | 字典未列枚举；按题意读取 |
| 40 | cgm_use_length | CGM 使用时长类别 | How long have you been doing this? | [C13](#c13) |
| 41 | typical_cgm_location | CGM 传感器通常放置位置 | Where do you usually place your CGM sensor? | [C14](#c14) |
| 42 | aid_use | 是否用过其他自动胰岛素输注系统 | Have you previously used any Automated Insulin Delivery Systems? | [C06](#c06) |
| 43 | aid_use_type___1 | 曾用AID：Medtronic 670G（独立多选项） | Medtronic 670G | [C10](#c10) |
| 44 | aid_use_type___2 | 曾用AID：OpenAPS（独立多选项） | OpenAPS | [C10](#c10) |
| 45 | aid_use_type___3 | 曾用AID：AndroidAPS（独立多选项） | AndroidAPS | [C10](#c10) |
| 46 | aid_use_type___4 | 曾用AID：其他（独立多选项） | Other | [C10](#c10) |
| 47 | why_start_loop___1 | 开始Loop的原因：减少低血糖次数/持续时间（独立多选项） | Reduce frequency/duration of hypoglycemic events | [C10](#c10) |
| 48 | why_start_loop___2 | 开始Loop的原因：改善睡眠（独立多选项） | Improve quality of sleep | [C10](#c10) |
| 49 | why_start_loop___3 | 开始Loop的原因：改善血糖/HbA1c（独立多选项） | Improve glucose control/HbA1c | [C10](#c10) |
| 50 | why_start_loop___4 | 开始Loop的原因：更多控制胰岛素输注（独立多选项） | Get more control over insulin delivery | [C10](#c10) |
| 51 | why_start_loop___5 | 开始Loop的原因：界面更易用（独立多选项） | More user friendly interface | [C10](#c10) |
| 52 | why_start_loop___6 | 开始Loop的原因：参与开源运动（独立多选项） | Participante in open source movement | [C10](#c10) |
| 53 | why_start_loop___7 | 开始Loop的原因：其他（独立多选项） | Other | [C10](#c10) |
| 54 | pump_type_use | Loop 配套泵型号 | What insulin pump do you use or will be using as part of Loop? | [C15](#c15) |
| 55 | what_cgm | Loop 配套 CGM 型号 | What CGM are you using or planning to use with Loop? | [C16](#c16) |
| 56 | additional_medications___1 | 额外药物：Symlin（独立多选项） | Symlin | [C10](#c10) |
| 57 | additional_medications___2 | 额外药物：二甲双胍（独立多选项） | Metformin | [C10](#c10) |
| 58 | additional_medications___3 | 额外药物：GLP-1类似物（独立多选项） | GLP-1 Analogs | [C10](#c10) |
| 59 | additional_medications___4 | 额外药物：DPP-4抑制剂（独立多选项） | DPP-4 Inhibitors | [C10](#c10) |
| 60 | additional_medications___5 | 额外药物：SGLT-2抑制剂（独立多选项） | SGLT-2 Inhibitors | [C10](#c10) |
| 61 | additional_medications___6 | 额外药物：草药/补充剂（独立多选项） | Herbs/Supplements | [C10](#c10) |
| 62 | additional_medications___7 | 额外药物：其他（独立多选项） | Other | [C10](#c10) |
| 63 | additional_medications___8 | 额外药物：无（独立多选项） | None | [C10](#c10) |
| 64 | lifetime_hypo_events | 自诊断 T1D 以来的严重低血糖事件自报次数 | SINCE YOU WERE DIAGNOSED WITH T1D, about how many severe hypoglycemic events have you experienced? | 字典未列枚举；按题意读取 |
| 65 | months_hypo_events | 过去3个月严重低血糖事件自报次数 | IN THE PAST 3 MONTHS, about how many severe hypoglycemic events have you experienced? | 字典未列枚举；按题意读取 |
| 66 | lifetime_dka_events | 自诊断 T1D 以来的酮症酸中毒自报次数 | SINCE YOU WERE DIAGNOSED WITH T1D, about how many episodes of Diabetic Ketoacidosis (DKA) have you experienced? | 字典未列枚举；按题意读取 |
| 67 | months_dka_events | 过去3个月酮症酸中毒自报次数 | IN THE PAST 3 MONTHS, about how many episodes of DKA have you experienced? | 字典未列枚举；按题意读取 |
| 68 | aware_low_blood | 开始发生低血糖时的感知程度：0从不感知，5始终感知 | On a scale of 0 to 5, with 0 representing never aware and 5 representing always aware how aware are you when you are beginning to experience low blood sugar (severe hypoglycemia)? | 字典未列枚举；按题意读取 |
| 69 | how_low_blood | 开始感觉低血糖症状时的血糖范围/从无症状 | How low does your blood sugar need to go before you feel symptoms? | [C17](#c17) |
| 70 | describe_health | 总体健康自评 | In general, how would you describe your health? | [C18](#c18) |
| 71 | activity_tracker | 使用/计划使用的活动追踪设备；不是该设备连续数据本身 | What activity tracking device, if any, are you currently using, or plan to use? | [C19](#c19) |
| 72 | exercise_30_min | 典型一周中每天至少运动30分钟的天数 | In a typical week, how many days do you spend at least 30 minutes doing any physical activities or exercise such as running, working out, yoga or pilates, aerobics, sports, gardening, PE in school, or walking for exercise? | 字典未列枚举；按题意读取 |
| 73 | aerobic_exercise | 典型一周中有氧运动天数 | How many days do you engage in aerobic exercise such as running, walking, swimming, biking, or using an elliptical trainer? | 字典未列枚举；按题意读取 |
| 74 | anaerobic_exercise | 典型一周中无氧运动天数 | How many days do you engage in anaerobic exercise such as weight lifting, sprints, interval training, or any rapid burst of hard exercise? | 字典未列枚举；按题意读取 |

## 已有Loop用户的基线补充

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 75 | loop_version | 所用 Loop 版本的枚举编码；数字不等于版本号 | Version of Loop currently in use. | [C20](#c20) |
| 76 | loop_branch | 所用 Loop 软件分支 | Which software branch are you currently using? | [C21](#c21) |
| 77 | typically_use_loop | Loop 典型使用时段：全天、仅夜间、部分日期 | How do you typically use Loop? | [C22](#c22) |
| 78 | who_share_cgm___0 | CGM分享对象：不分享（独立多选项） | I don't share | [C10](#c10) |
| 79 | who_share_cgm___1 | CGM分享对象：伴侣（独立多选项） | Partner | [C10](#c10) |
| 80 | who_share_cgm___2 | CGM分享对象：父母/监护人（独立多选项） | Parent/Guardian | [C10](#c10) |
| 81 | who_share_cgm___3 | CGM分享对象：其他家人（独立多选项） | Other Family | [C10](#c10) |
| 82 | who_share_cgm___4 | CGM分享对象：朋友（独立多选项） | Friend | [C10](#c10) |
| 83 | who_share_cgm___5 | CGM分享对象：其他（独立多选项） | Other | [C10](#c10) |
| 84 | how_share_cgm___0 | CGM分享方式：不分享（独立多选项） | I don't share | [C10](#c10) |
| 85 | how_share_cgm___1 | CGM分享方式：Nightscout（独立多选项） | Nightscout | [C10](#c10) |
| 86 | how_share_cgm___2 | CGM分享方式：Dexcom Share（独立多选项） | Dexcom Share | [C10](#c10) |
| 87 | how_share_cgm___3 | CGM分享方式：其他（独立多选项） | Other | [C10](#c10) |
| 88 | loop_apple_watch | 是否使用 Loop Apple Watch App | Do you use the Loop Apple Watch app? | [C06](#c06) |
| 89 | hcp_interaction_loop | 医护人员对本人使用 Loop 的知情/支持情况 | What  has been your interaction with your health care provider (HCP) regarding your use of Loop? | [C23](#c23) |

## 随访一般资料

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 90 | pregnant_f | 随访问项：当前是否怀孕；特别注意 2=否，不是0 | Are you currently pregnant? | [C04](#c04) |
| 91 | low_carb_diet_f | 随访问项：是否遵循低碳水饮食 | Do you follow a low-carbohydrate diet? | [C06](#c06) |
| 92 | use_insulin_pump | 随访时是否仍使用胰岛素泵 | Are you using an insulin pump? | [C06](#c06) |
| 93 | insulin_type_f | 随访问项：泵内使用的胰岛素制剂类别 | What type of insulin are you using in your pump? | [C09](#c09) |
| 94 | insulin_injection_f | 随访问项：是否在用泵之外还规律注射胰岛素；不是逐次注射剂量日志 | Do you regularly use insulin injections in addition to your insulin pump therapy? | [C06](#c06) |
| 95 | insulin_injection_reason_f___1 | 随访问项：额外注射原因：允许取下泵（独立多选项） | Allow removal of pump | [C10](#c10) |
| 96 | insulin_injection_reason_f___2 | 随访问项：额外注射原因：减少低血糖/DKA（原题选项，不是疗效认定）（独立多选项） | Reduce hypo/DKA | [C10](#c10) |
| 97 | insulin_injection_reason_f___3 | 随访问项：额外注射原因：胰岛素用量较高（原文措辞较简略）（独立多选项） | Insulin dose high enough | [C10](#c10) |
| 98 | insulin_injection_reason_f___4 | 随访问项：额外注射原因：运动/活动（独立多选项） | Sports/Activities | [C10](#c10) |
| 99 | insulin_injection_reason_f___5 | 随访问项：额外注射原因：泵故障（独立多选项） | Pump not working | [C10](#c10) |
| 100 | insulin_injection_reason_f___6 | 随访问项：额外注射原因：泵耗材用完（独立多选项） | Out of pump supplies | [C10](#c10) |
| 101 | insulin_injection_reason_f___7 | 随访问项：额外注射原因：其他（独立多选项） | Other | [C10](#c10) |
| 102 | afrezza_f | 随访问项：是否在用泵之外还规律使用 Afrezza 吸入胰岛素 | Do you regularly use Afrezza (inhaled insulin) in addition to your insulin pump therapy? | [C06](#c06) |
| 103 | afrezza_reason_f___1 | 随访问项：额外吸入胰岛素原因：餐时替代餐时Bolus（独立多选项） | Take at mealtime instead of mealtime bolus | [C10](#c10) |
| 104 | afrezza_reason_f___2 | 随访问项：额外吸入胰岛素原因：餐时在Bolus之外追加（独立多选项） | Take at mealtime in addition to mealtime bolus | [C10](#c10) |
| 105 | afrezza_reason_f___3 | 随访问项：额外吸入胰岛素原因：纠正高血糖时替代Bolus（独立多选项） | Take to correct highs instead of bolus | [C10](#c10) |
| 106 | afrezza_reason_f___4 | 随访问项：额外吸入胰岛素原因：纠正高血糖时在Bolus之外追加（独立多选项） | Take to correct highs in addition to bolus | [C10](#c10) |
| 107 | afrezza_reason_f___5 | 随访问项：额外吸入胰岛素原因：其他（独立多选项） | Other | [C10](#c10) |
| 108 | cgm_use | 随访时是否使用 CGM | Are you using a continuous glucose monitor (CGM)? | [C06](#c06) |
| 109 | additional_medications_f___1 | 随访问项：额外药物：Symlin（独立多选项） | Symlin | [C10](#c10) |
| 110 | additional_medications_f___2 | 随访问项：额外药物：二甲双胍（独立多选项） | Metformin | [C10](#c10) |
| 111 | additional_medications_f___3 | 随访问项：额外药物：GLP-1类似物（独立多选项） | GLP-1 Analogs | [C10](#c10) |
| 112 | additional_medications_f___4 | 随访问项：额外药物：DPP-4抑制剂（独立多选项） | DPP-4 Inhibitors | [C10](#c10) |
| 113 | additional_medications_f___5 | 随访问项：额外药物：SGLT-2抑制剂（独立多选项） | SGLT-2 Inhibitors | [C10](#c10) |
| 114 | additional_medications_f___6 | 随访问项：额外药物：草药/补充剂（独立多选项） | Herbs/Supplements | [C10](#c10) |
| 115 | additional_medications_f___7 | 随访问项：额外药物：其他（独立多选项） | Other | [C10](#c10) |
| 116 | additional_medications_f___8 | 随访问项：额外药物：无（独立多选项） | None | [C10](#c10) |
| 117 | aware_low_blood_f | 随访问项：开始发生低血糖时的感知程度：0从不感知，5始终感知 | On a scale of 0 to 5, with 0 representing never aware and 5 representing always aware how aware are you when you are beginning to experience low blood sugar (severe hypoglycemia)? | 字典未列枚举；按题意读取 |
| 118 | how_low_blood_f | 随访问项：开始感觉低血糖症状时的血糖范围/从无症状 | How low does your blood sugar need to go before you feel symptoms? | [C17](#c17) |

## 已有Loop用户的随访补充

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 119 | loop_versionf | 随访问项：所用 Loop 版本的枚举编码；数字不等于版本号 | Version of Loop currently in use. | [C24](#c24) |
| 120 | typically_use_loopf | 随访问项：Loop 典型使用时段：全天、仅夜间、部分日期 | How do you typically use Loop? | [C22](#c22) |
| 121 | pump_type_usef | 随访问项：Loop 配套泵型号 | What insulin pump do you use or will be using as part of Loop? | [C15](#c15) |
| 122 | who_share_cgmf___0 | 随访问项：CGM分享对象：不分享（独立多选项） | Who do you share your live CGM data with? (choice=I dont share my live CGM data with others.) | [C10](#c10) |
| 123 | who_share_cgmf___1 | 随访问项：CGM分享对象：伴侣（独立多选项） | Who do you share your live CGM data with? (choice=Partner) | [C10](#c10) |
| 124 | who_share_cgmf___2 | 随访问项：CGM分享对象：父母/监护人（独立多选项） | Who do you share your live CGM data with? (choice=Parent/Guardian) | [C10](#c10) |
| 125 | who_share_cgmf___3 | 随访问项：CGM分享对象：其他家人（独立多选项） | Who do you share your live CGM data with? (choice=Other family) | [C10](#c10) |
| 126 | who_share_cgmf___4 | 随访问项：CGM分享对象：朋友（独立多选项） | Who do you share your live CGM data with? (choice=Friend) | [C10](#c10) |
| 127 | who_share_cgmf___5 | 随访问项：CGM分享对象：其他（独立多选项） | Who do you share your live CGM data with? (choice=Other) | [C10](#c10) |
| 128 | how_share_cgmf___0 | 随访问项：CGM分享方式：不分享（独立多选项） | How do you share your live CGM data with others? (choice=I dont share my live CGM data with others.) | [C10](#c10) |
| 129 | how_share_cgmf___1 | 随访问项：CGM分享方式：Nightscout（独立多选项） | How do you share your live CGM data with others? (choice=Nightscout) | [C10](#c10) |
| 130 | how_share_cgmf___2 | 随访问项：CGM分享方式：Dexcom Share（独立多选项） | How do you share your live CGM data with others? (choice=Dexcom Share) | [C10](#c10) |
| 131 | how_share_cgmf___3 | 随访问项：CGM分享方式：其他（独立多选项） | How do you share your live CGM data with others? (choice=Other) | [C10](#c10) |
| 132 | loop_apple_watchf | 随访问项：是否使用 Loop Apple Watch App | Do you use the Loop Apple Watch app? | [C06](#c06) |
| 133 | hcp_interaction_loopf | 随访问项：医护人员对本人使用 Loop 的知情/支持情况 | What  has been your interaction with your health care provider (HCP) regarding your use of Loop? | [C25](#c25) |

## INSPIRE Survey (Adult)

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 134 | adult_aid_more_hopeful | 成人本人认为AID能对未来更有希望；属于态度/期望 | I will be more hopeful about my future with use of automated insulin delivery (AID). | [C26](#c26) |
| 135 | adult_aid_worry_less | 成人本人认为AID能减少对糖尿病的担忧；属于态度/期望 | I will worry less about diabetes with AID. | [C26](#c26) |
| 136 | adult_aid_reduce_fam_concern | 成人本人认为AID能减少家人对糖尿病的担忧；属于态度/期望 | AID will reduce my family's concerns about my diabetes. | [C26](#c26) |
| 137 | adult_aid_easy_todo_activities | 成人本人认为AID能更容易做想做的事；属于态度/期望 | AID will make it easier for me to do the things that I want to do without diabetes getting in the way. | [C26](#c26) |
| 138 | adult_aid_decrease_lows | 成人本人认为AID能减少低血糖；属于态度/期望 | AID will decrease how often I have low glucose levels. | [C26](#c26) |
| 139 | adult_aid_decrease_highs | 成人本人认为AID能减少高血糖；属于态度/期望 | AID will decrease how often I have high glucose levels. | [C26](#c26) |
| 140 | adult_aid_target_range | 成人本人认为AID能更多时间处于目标血糖范围；属于态度/期望 | AID will help me stay in my target range more often. | [C26](#c26) |
| 141 | adult_aid_improve_a1c | 成人本人认为AID能让HbA1c达到目标；属于态度/期望 | AID will improve my A1c to target level. | [C26](#c26) |
| 142 | adult_aid_eat_anytime | 成人本人认为AID能更容易在想吃时进食；属于态度/期望 | AID will make it easy to eat when I want. | [C26](#c26) |
| 143 | adult_aid_exercise_anytime | 成人本人认为AID能更容易在想运动时运动；属于态度/期望 | AID will make it easy to exercise when I want. | [C26](#c26) |
| 144 | adult_aid_mnge_work_school | 成人本人认为AID能更容易在工作/学校管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when I am at work or school. | [C26](#c26) |
| 145 | adult_aid_mnge_social_life | 成人本人认为AID能更容易在社交时管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when it comes to my social life/being with friends. | [C26](#c26) |
| 146 | adult_aid_mnge_sick_days | 成人本人认为AID能更容易在生病时管理糖尿病；属于态度/期望 | AID will help me manage sick days. | [C26](#c26) |
| 147 | adult_aid_sleep_better | 成人本人认为AID能睡得更好；属于态度/期望 | AID will help me sleep better. | [C26](#c26) |
| 148 | adult_aid_lows_nights | 成人本人认为AID能减少夜间低血糖；属于态度/期望 | I believe that I will have fewer lows during the night with AID. | [C26](#c26) |
| 149 | adult_aid_qual_life | 成人本人认为AID能改善生活质量；属于态度/期望 | AID will improve my overall quality of life. | [C26](#c26) |
| 150 | adult_aid_fam_qual_life | 成人本人认为AID能改善家人生活质量；属于态度/期望 | AID will improve my family's overall quality of life. | [C26](#c26) |
| 151 | adult_aid_mnge_drive_travel | 成人本人认为AID能更容易在驾车/旅行时管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when driving (for those who drive) or when traveling. | [C26](#c26) |
| 152 | adult_aid_mnge_sex_life | 成人本人认为AID能更容易在性生活方面管理糖尿病；属于态度/期望 | AID will help me manage diabetes when it comes to my sex life. | [C26](#c26) |
| 153 | adult_aid_mnge_drink_alcohol | 成人本人认为AID能若选择饮酒，更容易管理糖尿病；属于态度/期望 | AID will help me manage diabetes when I choose to drink alcohol. | [C26](#c26) |
| 154 | adult_aid_help_pregnancy | 成人本人认为AID能若怀孕，更容易管理糖尿病；属于态度/期望 | AID will help me if I am pregnant. | [C26](#c26) |
| 155 | adult_aid_longterm_comp | 成人本人认为AID能降低长期并发症风险；属于态度/期望 | AID will reduce my risk of long term complications. | [C26](#c26) |

## INSPIRE Survey (Youth)

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 156 | child_aid_more_hopeful | 青少年本人认为AID能对未来更有希望；属于态度/期望 | I will be more hopeful about my future with use of automated insulin delivery (AID). | [C26](#c26) |
| 157 | child_aid_worry_less | 青少年本人认为AID能减少对糖尿病的担忧；属于态度/期望 | I will worry less about diabetes with AID. | [C26](#c26) |
| 158 | child_aid_reduce_fam_concern | 青少年本人认为AID能减少家人对糖尿病的担忧；属于态度/期望 | AID will reduce my family's concerns about my diabetes. | [C26](#c26) |
| 159 | child_aid_easy_todo_activities | 青少年本人认为AID能更容易做想做的事；属于态度/期望 | AID will make it easier for me to do the things I want to do without diabetes getting in the way. | [C26](#c26) |
| 160 | child_aid_decrease_lows | 青少年本人认为AID能减少低血糖；属于态度/期望 | AID will decrease how often I have low glucose levels. | [C26](#c26) |
| 161 | child_aid_decrease_highs | 青少年本人认为AID能减少高血糖；属于态度/期望 | AID will decrease how often I have high glucose levels. | [C26](#c26) |
| 162 | child_aid_target_range | 青少年本人认为AID能更多时间处于目标血糖范围；属于态度/期望 | AID will help me stay in my target glucose range more often. | [C26](#c26) |
| 163 | child_aid_improve_a1c | 青少年本人认为AID能让HbA1c达到目标；属于态度/期望 | AID will improve my A1c to target level. | [C26](#c26) |
| 164 | child_aid_eat_anytime | 青少年本人认为AID能更容易在想吃时进食；属于态度/期望 | AID will make it easy to eat when I want. | [C26](#c26) |
| 165 | child_aid_exercise_anytime | 青少年本人认为AID能更容易在想运动时运动；属于态度/期望 | AID will make it easy to exercise when I want. | [C26](#c26) |
| 166 | child_aid_mnge_work_school | 青少年本人认为AID能更容易在工作/学校管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when I am at school or work. | [C26](#c26) |
| 167 | child_aid_mnge_social_life | 青少年本人认为AID能更容易在社交时管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when I am with my friends. | [C26](#c26) |
| 168 | child_aid_mnge_sick_days | 青少年本人认为AID能更容易在生病时管理糖尿病；属于态度/期望 | AID will help me manage sick days. | [C26](#c26) |
| 169 | child_aid_sleep_better | 青少年本人认为AID能睡得更好；属于态度/期望 | AID will help me sleep better. | [C26](#c26) |
| 170 | child_aid_lows_nights | 青少年本人认为AID能减少夜间低血糖；属于态度/期望 | I believe that I will have fewer lows during the night with AID. | [C26](#c26) |
| 171 | child_aid_qual_life | 青少年本人认为AID能改善生活质量；属于态度/期望 | AID will improve my overall quality of life. | [C26](#c26) |
| 172 | child_aid_fam_qual_life | 青少年本人认为AID能改善家人生活质量；属于态度/期望 | AID will improve my family's overall quality of life. | [C26](#c26) |

## INSPIRE Survey (Parent)

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 173 | parent_aid_reduce_fam_concern | 家长认为AID能帮助孩子减少家人对糖尿病的担忧；属于态度/期望 | AID will reduce my family's concerns about my child's diabetes. | [C26](#c26) |
| 174 | parent_aid_easy_todo_activities | 家长认为AID能帮助孩子更容易做想做的事；属于态度/期望 | AID will make it easier for my child to do what they want to do without diabetes getting in the way | [C26](#c26) |
| 175 | parent_aid_decrease_lows | 家长认为AID能帮助孩子减少低血糖；属于态度/期望 | AID will decrease how often my child has low glucose levels. | [C26](#c26) |
| 176 | parent_aid_decrease_highs | 家长认为AID能帮助孩子减少高血糖；属于态度/期望 | AID will decrease how often my child has high glucose levels. | [C26](#c26) |
| 177 | parent_aid_target_range | 家长认为AID能帮助孩子更多时间处于目标血糖范围；属于态度/期望 | AID will help my child stay in his/her target range more often. | [C26](#c26) |
| 178 | parent_aid_improve_a1c | 家长认为AID能帮助孩子让HbA1c达到目标；属于态度/期望 | AID will improve my child's A1c to target level. | [C26](#c26) |
| 179 | parent_aid_eat_anytime | 家长认为AID能帮助孩子更容易在想吃时进食；属于态度/期望 | AID will make it easy to eat when my child wants. | [C26](#c26) |
| 180 | parent_aid_exercise_anytime | 家长认为AID能帮助孩子更容易在想运动时运动；属于态度/期望 | AID will make it easy to exercise when my child wants. | [C26](#c26) |
| 181 | parent_aid_mnge_work_school | 家长认为AID能帮助孩子更容易在工作/学校管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when my child is at school or work. | [C26](#c26) |
| 182 | parent_aid_mnge_social_life | 家长认为AID能帮助孩子更容易在社交时管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when it comes to my child's social life/being with friends. | [C26](#c26) |
| 183 | parent_aid_mnge_sick_days | 家长认为AID能帮助孩子更容易在生病时管理糖尿病；属于态度/期望 | AID will help me manage my child's sick days. | [C26](#c26) |
| 184 | parent_aid_sleep_better | 家长认为AID能让家长自己睡得更好；属于态度/期望 | AID will help me sleep better. | [C26](#c26) |
| 185 | parent_aid_lows_nights | 家长认为AID能帮助孩子减少夜间低血糖；属于态度/期望 | I believe that my child will have fewer lows during the night with AID. | [C26](#c26) |
| 186 | parent_aid_qual_life | 家长认为AID能让家长自己改善生活质量；属于态度/期望 | AID will improve my overall quality of life. | [C26](#c26) |
| 187 | parent_aid_fam_qual_life | 家长认为AID能帮助孩子改善家人生活质量；属于态度/期望 | AID will improve my family's overall quality of life. | [C26](#c26) |
| 188 | parent_aid_mnge_travel | 家长认为AID能帮助孩子更容易在旅行时管理糖尿病；属于态度/期望 | AID will make managing diabetes easy when my child is driving (for those who drive) or when traveling. | [C26](#c26) |
| 189 | parent_aid_mnge_drink_alcohol | 家长认为AID能帮助孩子若选择饮酒，更容易管理糖尿病；属于态度/期望 | AID will help my child manage diabetes if she/he chooses to drink alcohol. | [C26](#c26) |
| 190 | parent_aid_help_pregnancy | 家长认为AID能帮助孩子若怀孕，更容易管理糖尿病；属于态度/期望 | AID will help my child manage diabetes if pregnant. | [C26](#c26) |
| 191 | parent_aid_longterm_comp | 家长认为AID能帮助孩子降低长期并发症风险；属于态度/期望 | AID will reduce my child's risk of long term complications. | [C26](#c26) |

## Diabetes Distress Scale

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 192 | not_skld_mang_dia_shld | 因自觉糖尿病管理能力不足而感到困扰的程度 | Feeling that I am not as skilled at managing diabetes as I should be. | [C27](#c27) |
| 193 | not_much_insul_shld | 因自觉未使用应有胰岛素量而感到困扰的程度 | Feeling that I am not taking as much insulin as I should. | [C27](#c27) |
| 194 | dnt_chk_bg_oft_shld | 因自觉血糖检查不够频繁而感到困扰的程度 | Feeling that I don't check my blood glucose level as often as I probably should. | [C27](#c27) |
| 195 | dnt_give_much_atten_shld | 因自觉关注糖尿病不够而感到困扰的程度 | Feeling I don't give my diabetes as much attention as I probably should. | [C27](#c27) |

## Diabetes Technology Attitudes Survey

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 196 | tech_made_life_better | 认同糖尿病技术改善了生活的程度 | Diabetes technology has made my life better | [C28](#c28) |
| 197 | tech_made_life_easier | 认同糖尿病技术让生活更轻松的程度 | Diabetes technology has made my life easier | [C28](#c28) |
| 198 | tech_made_health_better | 认同糖尿病技术改善健康的程度 | Diabetes technology has made my health better | [C28](#c28) |
| 199 | tech_more_good_than_bad | 认同糖尿病技术利大于弊的程度 | Diabetes technology does more good than bad | [C28](#c28) |
| 200 | tech_more_time_work_than_worth | 认同技术耗时费力超过其价值的程度；题意与前四项相反 | Diabetes technology takes more time and work than it is worth | [C28](#c28) |

## Pittsburgh Sleep Quality Index

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 201 | bed_time | 过去一个月通常的晚间上床时间 | During that past month, what time have you usually gone to bed at night? | 字典未列枚举；按题意读取 |
| 202 | min_to_fall_sleep | 过去一个月通常每晚入睡所需分钟数 | During the past month, how long (in minutes) has it usually taken you to fall asleep each night? | 字典未列枚举；按题意读取 |
| 203 | wake_up_time | 过去一个月通常的早晨起床时间 | During the past month, what time have you usually gotten up in the morning? | 字典未列枚举；按题意读取 |
| 204 | hours_sleep | 过去一个月通常每晚实际睡眠小时数；不是卧床时长 | During the past month, how many hours of actual sleep did you get at night? (This may be different than the number of hours you spent in bed.) | 字典未列枚举；按题意读取 |
| 205 | cant_sleep_in_30_min | 过去一个月因30分钟内不能入睡而出现睡眠困难的频率 | Cannot get to sleep within 30 minutes | [C29](#c29) |
| 206 | wake_up_middle_night | 过去一个月半夜/清晨醒来的频率 | Wake up in the middle of the night or early morning | [C29](#c29) |
| 207 | use_bathrm_night | 过去一个月夜间起床如厕的频率 | Have to get up to use the bathroom | [C29](#c29) |
| 208 | cant_breathe | 过去一个月呼吸不畅影响睡眠的频率 | Cannot breathe comfortably | [C29](#c29) |
| 209 | cough_snore_loudly | 过去一个月咳嗽/大声打鼾影响睡眠的频率 | Cough or snore loudly | [C29](#c29) |
| 210 | too_cold | 过去一个月太冷影响睡眠的频率 | Feel too cold | [C29](#c29) |
| 211 | too_hot | 过去一个月太热影响睡眠的频率 | Feel too hot | [C29](#c29) |
| 212 | bad_dreams | 过去一个月噩梦影响睡眠的频率 | Had bad dreams | [C29](#c29) |
| 213 | have_pain | 过去一个月疼痛影响睡眠的频率 | Have pain | [C29](#c29) |
| 214 | rate_overall_sleep_quality | 过去一个月总体睡眠质量：字典1很差至4很好；勿套用其他版本PSQI编码 | During the past month, how would you rate your sleep quality overall? | [C30](#c30) |
| 215 | take_sleep_meds | 过去一个月服用助眠药物的频率 | During the past month, how often have you taken medicine to help you sleep (prescribed or over the counter)? | [C29](#c29) |
| 216 | trouble_staying_awake | 过去一个月驾车、吃饭、社交时难以保持清醒的频率 | During the past month, how often have you had trouble staying awake while driving, eating meals, or engaging in social activity? | [C29](#c29) |
| 217 | keep_enough_enthusiasm | 过去一个月做事难以保持热情的问题程度 | During the past month, how much of a problem has it been for you to keep up enough enthusiasm to get things done? | [C31](#c31) |

## Hypoglycemia Fear Survey (Adult)

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 218 | not_rec_real_low | 担忧“未发现自己低血糖”的频率；不是事件实际发生次数 | Not recognizing/realizing I am having low blood sugar | [C32](#c32) |
| 219 | not_have_food_fruit | 担忧“身边没有食物、水果或果汁”的频率；不是事件实际发生次数 | Not having food, fruit, or juice with me | [C32](#c32) |
| 220 | pass_out_pub | 担忧“在公共场合昏倒”的频率；不是事件实际发生次数 | Passing out in public | [C32](#c32) |
| 221 | embarrass_soc_sit | 担忧“在社交中使自己或朋友尴尬”的频率；不是事件实际发生次数 | Embarrassing myself or my friends in a social situation | [C32](#c32) |
| 222 | have_react_alne | 担忧“独处时发生低血糖反应”的频率；不是事件实际发生次数 | Having a reaction while alone | [C32](#c32) |
| 223 | appear_stupid_drunk | 担忧“看起来迟钝或像喝醉”的频率；不是事件实际发生次数 | Appearing stupid or drunk | [C32](#c32) |
| 224 | lose_cntrl | 担忧“失去控制”的频率；不是事件实际发生次数 | Losing control | [C32](#c32) |
| 225 | no_one_arnd_help | 担忧“低血糖反应时身边无人帮助”的频率；不是事件实际发生次数 | No one being around to help me during a reaction | [C32](#c32) |
| 226 | have_react_driving | 担忧“驾车时发生低血糖反应”的频率；不是事件实际发生次数 | Having a reaction while driving | [C32](#c32) |
| 227 | mistake_have_accdnt | 担忧“犯错或发生事故”的频率；不是事件实际发生次数 | Making a mistake or having an accident | [C32](#c32) |
| 228 | bad_eval_criticize | 担忧“得到差评或受到批评”的频率；不是事件实际发生次数 | Getting a bad evaluation or being criticized | [C32](#c32) |
| 229 | diff_think_clear | 担忧“需要照顾他人时难以清晰思考”的频率；不是事件实际发生次数 | Difficulty thinking clearly when responsible for others | [C32](#c32) |
| 230 | lightheaded_dizzy | 担忧“头昏/眩晕”的频率；不是事件实际发生次数 | Feeling lightheaded or dizzy | [C32](#c32) |

## Hypoglycemia Fear Survey (Youth)

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 231 | not_recgnze_low | 担忧“未发现自己低血糖”的频率；不是事件实际发生次数 | Not recognizing that my blood sugar is low. | [C32](#c32) |
| 232 | not_have_food | 担忧“低血糖时身边没有食物、水果或果汁”的频率；不是事件实际发生次数 | Not having food, fruit, or juice with me when my blood sugar gets low. | [C32](#c32) |
| 233 | feel_dizzy | 担忧“因低血糖在公共场合眩晕或昏倒”的频率；不是事件实际发生次数 | Feeling dizzy or passing out in public because of low blood sugar. | [C32](#c32) |
| 234 | low_asleep | 担忧“睡眠时低血糖”的频率；不是事件实际发生次数 | Having low blood sugar while asleep. | [C32](#c32) |
| 235 | embrss_bc_low | 担忧“因低血糖使自己尴尬”的频率；不是事件实际发生次数 | Embarrassing myself because of low blood sugar. | [C32](#c32) |
| 236 | low_by_myself | 担忧“独处时低血糖”的频率；不是事件实际发生次数 | Having low blood sugar while I am by myself. | [C32](#c32) |
| 237 | look_stupid | 担忧“在他人面前显得迟钝或笨拙”的频率；不是事件实际发生次数 | Looking stupid or clumsy in front of other people. | [C32](#c32) |
| 238 | lose_ctrl | 担忧“因低血糖失控”的频率；不是事件实际发生次数 | Losing control because of low blood sugar. | [C32](#c32) |
| 239 | no_one_arnd | 担忧“低血糖时无人帮助”的频率；不是事件实际发生次数 | No one being around to help me during a low. | [C32](#c32) |
| 240 | mistake_school | 担忧“在学校犯错或发生事故”的频率；不是事件实际发生次数 | Making a mistake or having an accident at school. | [C32](#c32) |
| 241 | trble_school | 担忧“因低血糖期间发生的事在学校遇到麻烦”的频率；不是事件实际发生次数 | Getting in trouble at school because of something that happens when my sugar is low. | [C32](#c32) |
| 242 | have_seizure | 担忧“发生惊厥”的频率；不是事件实际发生次数 | Having seizures. | [C32](#c32) |
| 243 | lng_term_comp | 担忧“因低血糖产生长期并发症（受访者担忧内容）”的频率；不是事件实际发生次数 | Getting long-term complications from low blood sugar. | [C32](#c32) |
| 244 | dizzy_woozy | 担忧“低血糖时头昏”的频率；不是事件实际发生次数 | Feeling dizzy or woozy when my blood sugar is low. | [C32](#c32) |
| 245 | have_low_bld_sug | 担忧“发生低血糖”的频率；不是事件实际发生次数 | Having a low blood sugar. | [C32](#c32) |

## Hypoglycemia Fear Survey (Parent)

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 246 | child_not_recog_low | 担忧“孩子未发现自己低血糖”的频率；不是事件实际发生次数 | Child not recognizing/realizing that he/she is having a low. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 247 | child_not_have_food | 担忧“孩子身边没有食物、水果或果汁”的频率；不是事件实际发生次数 | Child not having food, fruit, or juice with him/her. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 248 | child_dizzy_passingout | 担忧“孩子在公共场合眩晕或昏倒”的频率；不是事件实际发生次数 | Child feeling dizzy or passing out in public. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 249 | child_low_while_sleep | 担忧“孩子睡眠时低血糖”的频率；不是事件实际发生次数 | Child having a low while asleep. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 250 | child_embarrass_self_others | 担忧“孩子在社交中使自己或亲友尴尬”的频率；不是事件实际发生次数 | Child embarrassing self or friends/family in a social situation. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 251 | child_have_low_alone | 担忧“孩子独处时低血糖”的频率；不是事件实际发生次数 | Child having a low while alone. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 252 | child_appear_stupid_clumsy | 担忧“孩子显得迟钝或笨拙”的频率；不是事件实际发生次数 | Child appearing to be stupid or clumsy. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 253 | child_lose_control_behavior | 担忧“孩子因低血糖行为失控”的频率；不是事件实际发生次数 | Child losing control of behavior due to low blood sugar. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 254 | no_one_help_child_during_low | 担忧“孩子低血糖时无人帮助”的频率；不是事件实际发生次数 | No one being around to help my child during a low. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 255 | child_make_mistakes | 担忧“孩子在学校犯错或发生事故”的频率；不是事件实际发生次数 | Child making a mistake or having an accident at school. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 256 | child_getting_a_bad_eval | 担忧“孩子因低血糖期间发生的事在学校获得差评”的频率；不是事件实际发生次数 | Child getting a bad evaluation at school because of something that happens when his/her sugar is low. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 257 | child_having_seizure | 担忧“孩子发生惊厥或抽搐”的频率；不是事件实际发生次数 | Child having seizures or convulsions. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 258 | child_longterm_complications | 担忧“孩子因频繁低血糖出现长期并发症（受访者担忧内容）”的频率；不是事件实际发生次数 | Child developing long term complications from frequent low blood sugar. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 259 | child_feel_faint | 担忧“孩子头昏或晕厥”的频率；不是事件实际发生次数 | Child feeling light-headed or faint. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |
| 260 | child_having_low | 担忧“孩子发生低血糖”的频率；不是事件实际发生次数 | Child having a low. | [C32](#c32)（来源：家长问卷PDF第2页，字典未列） |

## Hypoglycemia Confidence Scale

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 261 | when_exrcse | 对运动时应对低血糖的信心 | When you are exercising? | [C33](#c33) |
| 262 | when_sleep | 对睡眠时应对低血糖的信心 | When you are sleeping? | [C33](#c33) |
| 263 | when_drive | 对驾车时应对低血糖的信心 | When you are driving? | [C33](#c33) |
| 264 | when_soc_sit | 对社交时应对低血糖的信心 | When you are in social situations? | [C33](#c33) |
| 265 | when_alne | 对独处时应对低血糖的信心 | When you are alone? | [C33](#c33) |
| 266 | avd_prob_hypo | 对避免低血糖导致严重问题的信心 | Avoid serious problems due to hypoglycemia? | [C33](#c33) |
| 267 | catch_respnd | 对血糖降得太低之前识别并应对的信心 | Catch and respond to hypoglycemia before your blood sugars get too low? | [C33](#c33) |
| 268 | cont_despite_hypo | 对尽管有低血糖风险仍能从事想做活动的信心 | Continue to do the things you really want to do in your life, despite the risks of hypoglycemia? | [C33](#c33) |
| 269 | confdt_spouse | 估计伴侣对本人避免低血糖严重问题的信心；不是伴侣直接作答 | If you have a spouse or partner: What is your best guess about how confident your spouse or partner feels about your ability to avoid serious problems due to hypoglycemia? | [C33](#c33) |

## Technology Use for Diabetes Problem Solving

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 270 | pump_graphs_averg | 利用血糖仪/泵图表或平均值发现偏高偏低的频率 | I use graphs or averages from my meter or pump to help me see if I have been running too high or too low. | [C34](#c34) |
| 271 | search_online | 上网搜索血糖问题信息的频率 | I search for information online to help me with problems I am having with my blood sugars. | [C34](#c34) |
| 272 | text_another_person | 将血糖发短信给他人求助的频率 | I text my blood sugars to another person when I want to get help with problems I am having with highs or lows. | [C34](#c34) |
| 273 | talk_people_online | 在线与其他糖尿病患者交流求助的频率 | I go online to talk to other people with diabetes in order to get help with diabetes problems. | [C34](#c34) |
| 274 | calculate_bolus | 使用计算器/App/泵计算器计算胰岛素量的频率 | I use an actual calculator, a calculator app, or a bolus wizard on my pump to help me figure out how much insulin I should take. | [C34](#c34) |
| 275 | carb_counter_app_web | 使用App/网站估算食物碳水的频率 | I use a carbohydrate counter app, or website to help me figure out how much carbs are in the food I eat. | [C34](#c34) |
| 276 | set_alarms | 设置糖尿病相关事务提醒的频率 | I set alarms on my phone, pump, or meter to remind me to do things with my diabetes. | [C34](#c34) |
| 277 | see_graphs_after_selfcare | 改善自我管理后查看血糖图表/日志评估效果的频率 | I look at graphs or logs of my blood sugar numbers after I try to improve my self-care to see if it made a difference. | [C34](#c34) |
| 278 | contact_clinic_email_website | 通过邮件或患者网站联系诊所求助的频率 | I contact someone from the diabetes clinic through email or patient websites to get help with diabetes problems. | [C34](#c34) |

## Risk Taking Survey

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 279 | general_risk | 总体冒险意愿，0完全不愿至10非常愿意 | Are you in general a person who takes risk or do you try to avoid risks? | [C35](#c35) |
| 280 | finacial_risk | 财务冒险意愿；原始字段拼写 finacial 保留 | Are you a person who takes financial risk or do you try to avoid risks? | [C35](#c35) |
| 281 | career_risk | 职业冒险意愿 | Are you a person who takes career risk or do you try to avoid risks? | [C35](#c35) |
| 282 | health_risk | 健康相关冒险意愿 | Are you a person who takes health risk or do you try to avoid risks? | [C35](#c35) |

## Loopholes Survey

| 序号 | 原始字段 | 中文含义 | 官方题干 | 编码 / 字典范围 |
|---|---|---|---|---|
| 283 | nervous_about_loop | 开始使用 Loop 时的紧张程度 | How nervous were you to start using Loop? I was... | [C36](#c36) |
| 284 | trust_loop | 是否信任 Loop 正常工作 | Do you trust Loop is working correctly? | [C37](#c37) |
| 285 | number_weeks_to_trust_loop | 建立对 Loop 正常工作的信任所需周数 | How many weeks did it take to trust Loop was working correctly? | 字典未列枚举；按题意读取 |
| 286 | recommend_loop | 向其他 T1D 患者推荐 Loop 的可能性 | How likely are you to recommend Loop to another person with type 1 diabetes? | [C38](#c38) |
| 287 | not_use_loop | 是否认为某类 T1D 患者不适合用 Loop；不是本人是否停用 | Is there a kind of person with type 1 diabetes who should not use Loop? | [C37](#c37) |
| 288 | hard_start_loop | 开始使用 Loop 的困难程度 | How hard was it to start Loop? It was... | [C39](#c39) |
| 289 | help_starting_loop | 开始使用 Loop 时是否需要帮助 | Did you need any help starting Loop? | [C37](#c37) |
| 290 | who_helped_you_start_loop___1 | 开始Loop时的帮助者：已有Loop用户（独立多选项） | Who helped you start Loop?  (choice=Someone already using Loop) | [C10](#c10) |
| 291 | who_helped_you_start_loop___2 | 开始Loop时的帮助者：协助管理糖尿病的人（独立多选项） | Who helped you start Loop?  (choice=Someone who helps me with my diabetes) | [C10](#c10) |
| 292 | who_helped_you_start_loop___3 | 开始Loop时的帮助者：糖尿病医生/护士（独立多选项） | Who helped you start Loop?  (choice=My diabetes care doctor or nurse) | [C10](#c10) |
| 293 | who_helped_you_start_loop___4 | 开始Loop时的帮助者：糖尿病教育者（独立多选项） | Who helped you start Loop?  (choice=A diabetes educator) | [C10](#c10) |
| 294 | who_helped_you_start_loop___5 | 开始Loop时的帮助者：更懂计算机/技术的人（独立多选项） | Who helped you start Loop?  (choice=Someone who knows computers (or technology) better than me) | [C10](#c10) |
| 295 | read_about_loop_first | 开始使用前是否阅读过在线帖子/讨论 | Did you read any online posts or discussions about Loop before you started? | [C37](#c37) |
| 296 | loop_day_use | 是否曾在白天使用 Loop | Have you used Loop during the day? | [C37](#c37) |
| 297 | hard_use_during_day | 白天使用 Loop 的困难程度 | How hard is it to use Loop during the day? It is... | [C39](#c39) |
| 298 | well_work_during_day | 对白天运行效果的评价；1很好至5不好 | How well does Loop work during the day? It works... | [C40](#c40) |
| 299 | look_app_on_phone | 查看手机 Loop App 的频率 | How often do you look at the Loop app on your phone? | [C41](#c41) |
| 300 | tune_radio_frequency_day | 调整无线电频率的频率 | How often do you 'tune radio frequency'? | [C41](#c41) |
| 301 | hard_carb_entry | 使用碳水录入功能的困难程度 | How hard is it to use the carb entry on Loop? It is... | [C39](#c39) |
| 302 | what_carb_features___1 | 所用碳水录入功能：食物类型图标（独立多选项） | What features of carb entry do you use? Check all that apply. (choice=Food type (lollipop, taco, pizza images)) | [C10](#c10) |
| 303 | what_carb_features___2 | 所用碳水录入功能：手动录入其他食物（独立多选项） | What features of carb entry do you use? Check all that apply. (choice=Manual entry of other foods) | [C10](#c10) |
| 304 | what_carb_features___3 | 所用碳水录入功能：手动修改吸收时间（独立多选项） | What features of carb entry do you use? Check all that apply. (choice=Manually change absorption time) | [C10](#c10) |
| 305 | enter_carbs_dont_eat | 是否曾录入碳水但没有实际吃碳水；提示食物记录不一定是真实进食 | Do you ever put in carbs but do not eat any carbs? | [C37](#c37) |
| 306 | carbs_low_bg | 处理低血糖所吃碳水是否录入：1总是，2从不，3有时 | Do you enter carbohydrates you take for low blood sugar in Loop? | [C42](#c42) |
| 307 | manual_bolus_day | 是否使用过 Loop 手动 Bolus 功能 | Do you ever use the manual bolus on Loop? | [C37](#c37) |
| 308 | workout_glucose_target | 是否使用过运动血糖目标功能 | Do you ever use the Workout Glucose Targets on Loop? | [C37](#c37) |
| 309 | workout_glucose_target_hard | 使用运动血糖目标功能的困难程度 | How hard is it to use this feature? It is... | [C43](#c43) |
| 310 | workout_glucose_target_what___1 | 运动目标使用情境：运动/身体活动（独立多选项） | Workout/Physical Activity | [C10](#c10) |
| 311 | workout_glucose_target_what___2 | 运动目标使用情境：提高血糖目标以预防低血糖（独立多选项） | Want a high target glucose to prevent hypoglycemia | [C10](#c10) |
| 312 | workout_glucose_target_what___3 | 运动目标使用情境：饮酒且担心低血糖（独立多选项） | Consuming alcohol and worried about hypoglycemia | [C10](#c10) |
| 313 | loop_night_use | 是否曾在夜间使用 Loop | Have you used Loop during the night? | [C37](#c37) |
| 314 | well_work_during_night | 对夜间运行效果的评价；1很好至5不好 | How well does Loop work during the night? It works... | [C40](#c40) |
| 315 | manual_bolus_night | 是否在夜间手动给予 Bolus | Do you ever give a manual bolus in Loop at night? | [C37](#c37) |
| 316 | loop_connect_issue_frequency | Loop 绿圈变灰/红、未成功闭环的频率 | How often do you lose the green circle (it turns grey or red) meaning Loop is not successfully looping? | [C44](#c44) |
| 317 | manual_connection_fix | Loop 未成功运行时是否手动修复连接 | Do you manually try to fix the connection when Loop is not successfully Looping? | [C37](#c37) |
| 318 | manual_fix_wait_time | 通常等待多久才开始手动修复连接 | About how long do you usually wait before manually trying to fix the issue? | [C45](#c45) |
| 319 | how_manually_fix___1 | 手动修复方式：启动Dexcom App（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Launch Dexcom App) | [C10](#c10) |
| 320 | how_manually_fix___2 | 手动修复方式：用电源键软重启iPhone（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Soft restart iPhone using the power button) | [C10](#c10) |
| 321 | how_manually_fix___3 | 手动修复方式：组合键强制重启iPhone（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Hard restart iPhone using a combination of buttons that forces the phone off) | [C10](#c10) |
| 322 | how_manually_fix___4 | 手动修复方式：在Loop中断开并重连RileyLink（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Disconnect and reconnect RileyLink from phone using toggle in Loop) | [C10](#c10) |
| 323 | how_manually_fix___5 | 手动修复方式：关闭再开启RileyLink硬件（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Power cycle RileyLink hardware by turning it off and on) | [C10](#c10) |
| 324 | how_manually_fix___6 | 手动修复方式：更换RileyLink电池（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Replace battery in RileyLink) | [C10](#c10) |
| 325 | how_manually_fix___7 | 手动修复方式：完全退出Loop App（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Fully exit Loop app) | [C10](#c10) |
| 326 | how_manually_fix___8 | 手动修复方式：重装相同/新版Loop（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Reinstall same/updated Loop app) | [C10](#c10) |
| 327 | how_manually_fix___9 | 手动修复方式：重装相同/新版RileyLink软件（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Reinstall same/updated software on RileyLink) | [C10](#c10) |
| 328 | how_manually_fix___10 | 手动修复方式：关闭再开启iPhone蓝牙（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Turn iPhone Bluetooth off and on) | [C10](#c10) |
| 329 | how_manually_fix___11 | 手动修复方式：其他（独立多选项） | What things do you usually do to try and fix the issue? Check all that apply. (choice=Other) | [C10](#c10) |
| 330 | trouble_getting_supplies | 获取胰岛素、泵或 CGM 耗材是否有困难 | Have you had any trouble getting insulin, or pump and continuous glucose monitor (CGM) supplies, to use Loop? | [C37](#c37) |

## 全部答案编码

以下为官方字典的完整可选值，去除排版空行，不改代码。家长恐惧题的补充来源已逐行标注。无枚举不等于无约束，也不等于不存在缺失编码；本轮不转换任何原始值。

### C01

1=Male; 2=Female; 3=Non-binary

### C02

1=Hispanic or Latino; 2=Not Hispanic or Latino; 3=Do not wish to answer

### C03

1=White; 2=Black/African-American; 3=Asian; 4=Native Hawaiian/Other Pacific Islander; 5=American Indian/Alaskan Native; 6=Prefer not to answer; 7=More than one race

### C04

1=Yes; 2=No; 3=Do not wish to answer

### C05

1=Less than 3 months ago; 2=3 to less than 6 months ago; 3=6 to less than 12 months ago; 4=Over 12 months ago

### C06

1=Yes; 0=No

### C07

0=Never used a pump before; planning to start with Loop; 1=Previously used a pump, but not currently using one; planning to start with Loop; 2=Currently using a pump

### C08

1=Less than 3 months; 2=3 to less than 6 months; 3=6 months to less than 1 year; 4=1 year to less than 2 years; 5=2 years to less than 5 years; 6=5 or more years

### C09

1=Humalog (Lispro); 2=Novolog (Aspart); 3=Apidra (Glusine); 4=Fiasp (Rapid Aspart); 5=Regular insulin

### C10

1=Checked; 0=Unchecked

### C11

1=Use a bolus calculator; 2=Count carbohydrates and use insulin to carb ratios; 3=Base the mealtime insulin (bolus) amount on experience; 4=Other

### C12

0=Never used a CGM before; planning to start with Loop; 1=Previously used a CGM, but not currently wearing one regularly; planning to start with Loop; 2=Currently wearing a CGM regularly

### C13

1=Less than 3 months; 2=3 months to less than 6 months; 3=6 months to less than 1 year; 4=1 year to less than 2 years; 5=2 years to less than 5 years; 6=5 or more years

### C14

1=Belly (abdomen); 2=Buttocks; 3=Arm; 4=Other location

### C15

1=MM 515/715; 2=MM 522/722; 3=MM 523/723; 4=MM 554/754; 5=Omnipod

### C16

1=Dexcom G4; 2=Dexcom G5; 3=Dexcom G6; 4=Medtronic Enlite; 5=Abbott Libre

### C17

1=At least 70 mg/dL; 2=60-69 mg/dL; 3=50-59 mg/dL; 4=40-49 mg/dL; 5=Less than 40 mg/dL; 6=I never feel symptoms when my blood sugar is low

### C18

5=Excellent; 4=Very Good; 3=Good; 2=Fair; 1=Poor

### C19

0=None; 1=Apple Watch; 2=Fitbit; 3=Other

### C20

18=v1.9.6; 17=v1.9.5; 16=v1.9.4; 15=v1.9.3; 14=v1.9.2; 13=v1.9.1; 12=v1.9; 11=v1.5.6; 10=v1.5.4; 9=v1.5.2; 8=v1.5.1; 7=v1.5.0; 6=v1.4.0; 5=v1.3.3; 4=v1.3.2; 3=v1.3.1; 2=I'm using a newer version than is listed above.; 1=I'm using an older version than is listed above.

### C21

0=I don't know.; 1=Master branch; 2=Dev branch; 3=A custom branch

### C22

1=24/7; 2=Night only; 3=Only some days

### C23

1=HCP is not aware I am using Loop; 2=HCP is aware but will not discuss with me; 3=HCP is supportive and assists me in Loop management; 4=Other

### C24

19=v2.0; 18=v1.9.6; 17=v1.9.5; 16=v1.9.4; 15=v1.9.3; 14=v1.9.2; 13=v1.9.1; 12=v1.9; 11=v1.5.6; 10=v1.5.4; 9=v1.5.2; 8=v1.5.1; 7=v1.5.0; 6=v1.4.0; 5=v1.3.3; 4=v1.3.2; 3=v1.3.1; 2=I'm using a newer version than is listed above.; 1=I'm using an older version than is listed above.

### C25

1=HCP is not aware I am using Loop.; 2=HCP is aware but will not discuss with me.; 3=HCP is supportive and assists me in Loop management.; 4=Other

### C26

0 = N/A; 1 = Strongly Disagree; 2 = Disagree; 3 = Neither Agree nor Disagree; 4 = Agree; 5 = Strongly Agree

### C27

1= Not a problem; 2=A slight problem; 3=A moderate problem; 4=A somewhat serious problem; 5=A serious problem; 6=A very serious problem

### C28

1 = Strongly disagree; 2 = Disagree; 3 = Neutral; 4 = Agree; 5 = Strongly agree

### C29

0 = Not during the past month; 1 = Less than once a week; 2 = Once or twice a week; 3 = Three or more times a week

### C30

1 = Very bad; 2 = Fairly bad; 3 = Fairly good; 4 = Very good

### C31

0 = No problem at all; 1 = Only a very slight problem; 2 = Somewhat of a problem; 3 = A very big problem

### C32

0 = Never; 1 = Rarely; 2 = Sometimes; 3 = Often; 4 = Almost always

### C33

1 = Not confident at all; 2 = A little confident; 3 = Moderately confident; 4 = Very confident

### C34

0 = Never; 1 = Once every few months; 2 = Once a month; 3 = Once a week; 4 = A few times a week; 5 = Every day

### C35

0 = Not at all prepared to take risks; 1; 2; 3; 4; 5; 6; 7; 8; 9; 10 = Very much prepared to take risks

### C36

1 = Not at all nervous; 2; 3 = A little nervous; 4; 5 = Very nervous

### C37

1 = Yes; 0 = No

### C38

1 = Not likely; 2; 3 = Likely; 4; 5 = Very likely

### C39

1 = Easy; 2; 3; 4; 5 = Hard

### C40

1 = Very Well; 2; 3 = Alright; 4; 5 = Not Well

### C41

5 = Every hour; 4 = Every few hours; 3 = Once a day; 2 = Once a week; 1 = Hardly ever

### C42

1 = Always; 2 = Never; 3 = Sometimes

### C43

1 Easy; 2; 3; 4; 5 Hard

### C44

5 = Every hour; 4 = Every few hours; 3 = Once a day; 2 = Once a week; 1 = Hardly ever; 0 = Never

### C45

1 = Less than 15 minutes; 2 = 15 to 30 minutes; 3 = More than 30 minutes

## 版本差异

`typical_cgm_locationf` 在官方字典中有定义，但本包 Surveys.txt 实际没有这一列。本说明不把它计入330列，也不从其他列生成它。`finacial_risk` 是原始拼写，保持不改。
