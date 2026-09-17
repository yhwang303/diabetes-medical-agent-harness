DCLP3 Protocol – Public Dataset

Link to public website:  https://public.jaeb.org/datasets/diabetes
Protocol details: see full protocol PDF, included with this dataset
Primary Manuscript: https://www.nejm.org/doi/full/10.1056/NEJMoa1907863
Table of Contents – Data Files, sorted by Source, then Data File Name
Note: this revised version of the dataset, dated 04 Aug 2022, includes the age of each participant at the time of enrollment.
Data File Name
Source
Description
Ketone
Abbott Precision Xtra ketone meter
List of ketone measurements obtained during the study
AdvEvent
CRF
Adverse events form
DeviceIssue
CRF
Device Issue Form
DiabDKAEvent
CRF
Table of additional data collected in association with reported Hyperglycemia or DKA Events
DiabLocalHbA1c
CRF
Local HbA1c form
DiabMedHistory
CRF
Medical History Updates form
DiabPhysExam
CRF
Collects Diabetes Physical Exam Form data
DiabPregnancyTest
CRF
Diabetes Pregnancy Test form
DiabScreening 
CRF
Collects Diabetes Screening Form data
DiabSocioEcon
CRF
Diabetes Socioeconomic Information form
DiabUnschedContact
CRF
Diabetes unscheduled contact form
FollowUpCTV
CRF
List of Follow-Up Complete-the-Visit forms
Insulin
CRF
List of all types of insulins for patient
InsulinPumpSettings
CRF
Insulin Therapy form
MedicalCondition
CRF
Medical Condition Form
Medication
CRF
Medications Form
PtFinalStat
CRF
Final Status Form
PtRoster
CRF
List of study participants
QCTest
CRF
Device quality control testing for the Blood Glucose Meter and the Blood Ketone Meter
RandomizationCTV
CRF
Randomization Complete the Visit Form
RunInReview
CRF
List of Run-In Review data 
ScreeningCTV
CRF
Screening Complete the Visit Form
Training
CRF
List of Study Device Training forms
VisitInfo
CRF
Visit Information Form
DexcomClarityCGM
Dexcom Clarity
CGM readings from Dexcom Clarity files
DexcomClarityMeter
Dexcom Clarity
Calibration events recorded in Dexcom Clarity files
cgm
Dexcom Clarity, Tandem pump, and others
aggregation of CGM data from multiple sources within the study that was used for the analyses presented in the primary manuscript
gluIndices
Dexcom Clarity, Tandem pump, and others
participant-level glycemic indices calculated under several assumptions; it uses the previous cgm dataset as input
DiasendCGM
Diasend
Readings from CGM tab of Diasend file
OtherCGM
Miscellaneous sources
List of CGM readings from other miscellaneous sources
SampleResults
Central lab
Central lab HbA1c and C-peptide sample results data
clarkeHypoAwarness
Participant questionnaire
Clark Hypoglycemia Awareness Questionnaire
diabetesDistressAdult
Participant questionnaire
Diabetes Distress Questionnaire - adult version
diabetesDistressParent
Participant questionnaire
Diabetes Distress Questionnaire - teen’s parent version
diabetesPersonality
Participant questionnaire
Diabetes Specific Personality Questionnaire
hyperglycemiaAvoidance
Participant questionnaire
Hyperglycemia Avoidance Scale
hypoglycemiaConfidence
Participant questionnaire
Hypoglycemia Confidence Scale
hypoglycemiaFearAdult
Participant questionnaire
Hypoglycemia Fear Survey – adult version
hypoglycemiaFearTeen
Participant questionnaire
Hypoglycemia Fear Survey – teen version
hypoglycemiaFearTeenParent
Participant questionnaire
Hypoglycemia Fear Survey – teen’s parent version
inspireAdult
Participant questionnaire
INSPIRE Survey- adult version
inspireTeen
Participant questionnaire
INSPIRE Survey- teen version
inspireTeenParent
Participant questionnaire
INSPIRE Survey- teen’s parent version
systemUsability
Participant questionnaire
System Usability Scale
technologyAcceptance
Participant questionnaire
Technology Acceptance
technologyExpectations
Participant questionnaire
Technology Expectations Survey
RocheMeter
Roche Accu-Chek Guide blood glucose meter
List of BG meter readings from Roche file
Pump_BasalRateChange
Tandem pump
One record per basal rate change due to pumping events
Pump_BolusDelivered
Tandem pump
One record per bolus delivered by pump
Pump_CGMGlucoseValue
Tandem pump
One record per CGM glucose value recorded on the pump



Ketone -- List of ketone measurements obtained during the study
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



PtID
Participant ID



DataDtTm
Ketone date-time of measurement



DeviceModel
Device used to measure the ketone



Ketone
Ketone measurement value (mmol/L)



IsQCtest
Whether the results are from a ketone test (0) or a control solution (1)



IsQCtestJaeb
Whether a reading is test(0) or control(1) -- this column may contain adjustments made at Jaeb



DataDtTm_adjusted
Local time of event, including adjustments made at Jaeb




AdvEvent -- Adverse Event Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient Identifier



AENotifiedDt
Date notified of/identified adverse event



ParentMedCondListID
Medical Condition as coded by study investigator



ParentMedCondListIDAdmin
Medical Condition as coded by study Medical Monitor



AEOnsetDt
Date of onset (or worsening of a pre-existing condition)



AEPrEnroll
Is the adverse event a worsening of a pre-existing condition present prior to study entry


Yes, No
AENotedStdyVisExam
Was the adverse event an abnormality (or worsening of an existing abnormality) identified on a study visit exam


Yes, No
AEIntensity
Maximum intensity (Severity)


Mild, Moderate, Severe
AERelStdyTrt
Is there a reasonable possibility that the event was caused by a study treatment/study device


Yes, No
AERelStdyTrtUncertain
Uncertain which treatment/study device caused event


1 => Checked
AERelStdyProc
Is there a reasonable possibility that the event was caused by a study procedure


Yes, No
AEEffectTrt
Effect on study treatment/device


No change, Discontinued temporarily, Discontinued permanently, Reduced dose => If study treatment is medication, reduced dose, Reduced use frequency/schedule
AESerious
Does the event meet criteria for a serious adverse event


Yes, No
AETrt
Did patient receive treatment for the Adverse Event


Yes, No
AESurg
Surgery/procedure


Yes, No
AESurgDt
Date of surgery/procedure



AEMedsList
If yes, list medications here



AEOthTrt
Other Treatment of Adverse Event


Yes, No
AEOutcome
Outcome


Ongoing (further improvement / worsening possible) => Ongoing (further improvement or worsening possible), Ongoing, medically stable => Ongoing, medically stable (further change not expected), Complete Recovery, Recovered with Sequelae, Fatal
AEResDt
Date of Recovery (with or without sequelae)



AEDeathCause
Cause of death



AEDeathDt
Date of Death



AEDeath
Criteria Defining Event as Serious Adverse Event: Death


1 => Checked
AEConAnomaly
Criteria Defining Event as Serious Adverse Event: Congenital Anomaly


1 => Checked
AELifeThreat
Criteria Defining Event as Serious Adverse Event: Life Threatening


1 => Checked
AEHosp
Criteria Defining Event as Serious Adverse Event: Hospitalization -- inital or prolonged


1 => Checked
AEDisability
Criteria Defining Event as Serious Adverse Event: Significant Disability or Incapacity


1 => Checked
AEOther
Criteria Defining Event as Serious Adverse Event: Other


1 => Checked
Weight
Weight
0


WeightMeas
Weight Measurement


lbs, kgs
WeightNotAvail
Weight: Not available


1 => Checked
AERelLabData
Relevant Tests/Laboratory Data


Yes, No
AEOthRelHx
Other relevant history, including preexisting medical conditions (e.g., allergies, pregnancy, smoking and alcohol use, hepatic/renal dysfunction, etc)


Yes, No
AEMedProd
Concomitant medical products and therapy dates (exclude treatment of event)


Yes, No
MMAERelStdyTrt
MM Is there a reasonable possibility that the event was caused by a study treatment/study device


Yes, No, Unrelated, Unlikely related, Possibly related, Probably related, Definitely related, Not assessable
MMAESerious
MM Does the event meet criteria for a serious adverse event


Yes, No
MMUnexpected
MM Was the event unexpected 


Yes, No
AERelStdyDrugDevice
What is the relationship of the event to study drug/device


Unrelated, Unlikely related, Possibly related, Probably related, Definitely related, Not assessable
AERelStdyDrugDeviceUncertain
Uncertain which study drug/device is related to event


1 => Checked
MMHospDiscRptObtained
MM Hospital Discharge Reports Obtained


Yes, No, Not Requested, No, Requested Not Obtained
AERelStdyTrtHighLvl
If Yes, to which of the following was the adverse event possibly related


Study drug (drug effect) => Study drug (drug effect) or biological product, Other study treatment => Other study treatment (e.g., laser, surgical procedure), Study device (function), Administration of study drug/device => Administration of study drug/device (e.g., injection site bleeding), Prep for the administration of study drug, Study diagnostic procedure => Study examination or testing procedure (not part of the intervention), More than one of the above
AERelStdyTrtWhich
If related to the study drug/device and the study has more than one study drug or device, please indicate which one or indicate uncertain if you cannot determine



AERelStdyDrugDeviceHighLvl
If possibly, probably or definitely related, to which of the following was the adverse event possibly/probably/definitely related


Study drug (drug effect) => Study drug (drug effect) or biological product, Other study treatment => Other study treatment (e.g., laser, surgical procedure), Study device (function), Administration of study drug/device => Administration of study drug/device (e.g., injection site bleeding), Prep for the administration of study drug, Study diagnostic procedure => Study examination or testing procedure (not part of the intervention), More than one of the above
AERelStdyDrugDeviceWhich
If related to the study drug/device and the study has more than one study drug or device, please indicate which one or indicate uncertain if you cannot determine



MMAERelStdyTrtHighLvl
MM To which of the following was the adverse event possibly related


Study drug (drug effect) => Study drug (drug effect) or biological product, Other study treatment => Other study treatment (e.g., laser, surgical procedure), Study device (function), Administration of study drug/device => Administration of study drug/device (e.g., injection site bleeding), Prep for the administration of study drug, Study diagnostic procedure => Study examination or testing procedure (not part of the intervention), More than one of the above
MMCodingForAnalysis
Category of medical condition not available in MedDRA list; assigned by medical monitor





DeviceIssue -- Device Issue Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient ID



InvDevice
Investigated Device



DevIssueType
Type of device issue


Device malfunction, User error => Participant user error, Study staff user error, Inadequate instructions / training, Inadequate labeling, Other
DevIssueOnsetDt
Date problem first occurred/was identified



DevIssueLocation
Location of Occurrence


Home, Inpatient, Clinic (outpatient), Other
DevIssueFrequency
Frequency


Single Event, Intermittent, Continuous
DevIssueEffect
Effect on study treatment/device


No change, Study device modified/adjusted, Study device replaced, Discontinued temporarily, Discontinued permanently
DevIssueReplacedDt
Date replaced or modified device first used by participant (leave empty if not applicable or not yet used by subject)



DevIssueAE
Is the device issue related to an adverse event


Yes, No
DevIssueAELikely
If no, please describe the likelihood that the device issue could have led to an adverse event


Not assessable, Not possible, Unlikely, Possibly, Probably, Certainly
DevIssueUADE
Can the device deficiency or issue be classified as an Unanticipated Adverse Device Effect (UADE)


Yes, No, Uncertain
DevIssueAELikelyHypo
What adverse event could have occurred: Hypoglycemia


1 => Checked
DevIssueAELikelyHyper
What adverse event could have occurred: Hyperglycemia


1 => Checked
DevIssueAELikelyOth
What adverse event could have occurred: Other


1 => Checked


DiabDKAEvent -- Table of additional data collected in association with reported Hyperglycemia or DKA Events
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



DKAOccurDt
Date of Event



DKAOccurDtApprox
Date of event is approximate


1 => Checked
DKAOccurDtUnk
Unknown: Date of event


1 => Checked
DKAMetCriteria
Did event meet study criteria listed above for DKA


Definitely, Probably => Probably, based on available information, No => No (i.e., Hyperglycemia event but not DKA), Cant deter => Cannot determine from the available information
GlucLevel
Glucose level



GlucLevelUnits
Glucose Units


mg/dL, mmol/L
GlucLevelUnk
Unknown: Glucose level


1 => Checked
KetoneResUnk
Unknown: Ketone Result


1 => Checked
KetoneSerumUnits
Ketone result: Serum units


mg/dL, mmol/L
KetoneUrine
Ketone result: Urine


Negative, Small, Medium, Large, Extra Large
HCO3
HCO3



HCO3Unk
Unknown: HCO3


1 => Checked
pH
pH



pHSample
pH Sample


Arterial blood, Venous
pHUnk
Unknown: pH


1 => Checked
BUN
BUN



BUNUnk
Unknown: BUN


1 => Checked
CerebEdema
Symptomatic cerebral edema


Yes, No, Unknown
EventCauseStdyDev
Is there any evidence that a study device (e.g., blood glucose meter and/or HHM) contributed to the event (either device malfunction or improper use by user)


Yes, No
EventCauseNonStdy
Is there any indication of non-study-device-related factors that contributed to the occurrence of the event


Yes, No
DKAOutcome
Outcome


Fully recovered, Other, Unknown
SensorWear
Was the participant wearing a CGM sensor at the time of the event


Yes, No, Unknown
SensorGluc
CGM sensor glucose reading
0.00
500.00

SensorGlucUnits
CGM sensor glucose units


mg/dL, mmol/L
SensorGlucUnk
Unknown: CGM sensor glucose reading


1 => Checked
AutoInsDelivWear
Was the participant wearing an automated insulin delivery system at the time of the event


Yes, No, Unknown
AutoInsDelivMode
If Yes, was the system in auto mode or manual mode


Auto Mode, Manual Mode, Unknown


DiabLocalHbA1c -- Local HbA1c form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient Identifier



Visit
Visit



HbA1cNotDone
Local HbA1c not done


1 => Checked
HbA1cTestDt
Date of HbA1c test



HbA1cTestMeth
Method of testing


DCA point of care, Afinion point of care, Other point of care, Lab, Unknown
HbA1cTestRes
HbA1c results
4.0
15.0



DiabMedHistory -- Medical History Updates form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient Identifier



SHSinceLastVis
Reportable Hypoglycemic event


Yes, No
DKASinceLastVis
Definite or Probable Reportable Severe Hyperglycemic or DKA Event


Yes, No
OthAESinceLastVis
Other reportable adverse event or adverse device effect (ADE)


Yes, No
MedCondSinceLastVis
Did the participant report a new medical condition that does not meet the definition of a reportable adverse event and has not previously been recorded on the Medical Conditions Form


Yes, No
DevProbSinceLastVis
Did the participant report having any reportable device  problems while using a study device since the last contact


Yes, No
MedSinceLastVis
Did the participant report any changes or new medications since the last contact


Yes, No
InsulinSinceLastVis
Did the participant report any changes in insulin type or insulin delivery method since the last contact


Yes, No
Visit
Visit





DiabPhysExam -- Collects Diabetes Physical Exam Form data
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient identifier



Visit
Visit



PhysExamNotDone
No physical exam performed


1 => Checked
Weight
Weight
1


WeightUnits
Weight units


lbs, kg
WeightUnk
Weight not measured


1 => Checked
Height
Height
1


HeightUnits
Height units


in, cm
HeightUnk
Height not measured


1 => Checked
BldPrSys
Blood pressure systolic
60
300

BldPrDia
Blood pressure diastolic
30
160

BldPrUnk
Blood pressure not measured


1 => Checked
PEHeartRt
Heart rate bpm
40
170

PEHeartRtUnk
Heart rate not measured


1 => Checked
Temp
Temperature
34.0
106.0

TempUnits
Temperature units


Celsius, Fahrenheit
TempUnk
Temperature not measured


1 => Checked
FingStkBG
Fingerstick blood glucose result
0.0


FingStkBGUnits
Fingerstick blood glucose result units


mg/dL, mmol/L
FingStkBGUnk
Fingerstick blood glucose not measured


1 => Checked
PEAbnormal
Were any clinically significant abnormalities found


Yes, No
PhysExamPerf
Was a physical exam performed


Yes, No


DiabPregnancyTest -- Diabetes Pregnancy Test form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



PregTestDt
Date of pregnancy test



PregTestResult
Pregnancy test result


Positive, Negative
PregTestNotDone
Preg test not done


1 => Checked
PregTestNotDoneReas
Preg test not done reason


Subject is male => Participant is male, Surgically sterile, Post-menopausal, Pre-pubertal, Other
PregTestBrand
Brand of test used



PregTestLotNum
Lot number



PregTestType
Test type performed


Urine, Blood
PregTestMaterial
Test materials used


OTC Kit, Local Lab
PregTestExpDt
Date OTC test expires





DiabScreening -- Collects Diabetes Screening Form data
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient identifier



EligCritMet
Eligibility criteria has been met by the participant


1 => Checked
ExclCritAbsent
Exclusion criteria is absent for the participant


1 => Checked
AgeAtEnrollment
Age at Enrollment



Gender
Gender


M => Male, F => Female
Ethnicity
Ethnicity


Hispanic or Latino, Not Hispanic or Latino, Unknown/not reported
Race
Race


White, Black/African American, Asian, Native Hawaiian/Other Pacific Islander, American Indian/Alaskan Native, More than one race, Unknown/not reported
RaceDs
If More than one race selected please specify



DiagDt
Date of Diagnosis



DiagDtApprox
Date of diagnosis of diabetes is Approximate


1 => Checked
DiagDtUnk
Date of diagnosis of diabetes is Unknown


1 => Checked
DiagAge
Age at diagnosis
0
99

DiagAgeApprox
Age at diagnosis is Approximate


1 => Checked
DiagAgeUnk
Age at diagnosis is Unknown


1 => Checked
SHMostRecent
Estimate of when most recent severe hypoglycemic event occurred


Never, < 3 months ago, 3-<6 months ago, 6-12 months ago, More than 12 months ago, 6-<12 months ago, 1-<5 years ago, 5-<10 years ago, >=10 years ago
SHNumLast12Months
Estimated number of  severe hypoglycemic events in the last 12 months


0, 1, 2, 3, 4, 5-10, >=10
SHSeizComa
Estimated number of severe hypoglycemic events involving seizure/coma


0, 1, 2, 3, 4, 5-10, >=10
SHSeizComaLast12Months
Number of severe hypoglycemic events involving seizure/coma in last 12 months


0, 1, 2, 3, 4, 5-10, >=10
DKAMostRecent
Estimate of when most recent definite or probable DKA event occurred


Never, < 3 months ago, 3-<6 months ago, 6-12 months ago, More than 12 months ago, 6-<12 months ago, 1-<5 years ago, 5-<10 years ago, >=10 years ago
DKANumLast12Months
Estimated number of definite or probable DKA events in the last 12 months


0, 1, 2, 3, 4, 5-10, >=10
InsModPump
Insulin modality: pump


1 => Checked
InsModInjections
Insulin modality: Injections


1 => Checked
InsModInhaled
Insulin modality: Inhaled


1 => Checked
InsModNone
Insulin modality: None


1 => Checked
PumpUse
How long has participant been using an insulin pump


<3 months, 3-<6 months, 6 months-<1 year, 1-<2 years, 2-<5 years, >= 5 years
PumpType
Manufacturer of pump



PumpTypeUnk
Manufacturer of pump Unknown


1 => Checked
UnitsInsTotal
Total daily insulin in units
1
300

UnitsInsUnk
Total daily insulin in units Unknown


1 => Checked
UnitsInsBasilOrLongAct
Total daily basal or long acting insulin units
1
200

UnitsInsBasilOrLongActUnk
Total daily basal or long acting insulin units Unknown


1 => Checked
NumPumpBolusOrShortAct
Number of boluses, injections, or inhaled doses per day
0
50

NumPumpBolusOrShortActUnk
Number of boluses, injections, or inhaled doses per day Unknown


1 => Checked
BGTestAvgNumMeter
Average number of BG tests from meter
0
30

BGTestMetDatNotAvail
Average number of BG tests from meter not available


1 => Checked
BGTestAvgNumPtRep
Average number of BG tests from participant
0
30

BGTestPtRepNotAvail
Average number of BG tests from participant not available


1 => Checked
CGMUseStat
Status of CGM use in last month


Never, In past, but not current, Current
CGMUseDur
How long has the participant been using CGM


<3 months, 3-<6 months, 6 months-<1 year, 1-<2 years, 2-<5 years, >= 5 years
CGMUseDevice
Which CGM device is being used


Abbott, Dexcom, Medtronic, Senseonics
CGMUseLast1Month
In the past month, how many days has the participant used the CGM
1
30

CGMUseLast1MonthUnk
In the past month, how many days has the participant used the CGM Unknown


1 => Checked
PhysExamNotDone
No physical exam performed


1 => Checked
Weight
Weight
1


WeightUnits
Weight units


lbs, kg
WeightUnk
Weight not measured


1 => Checked
Height
Height
1


HeightUnits
Height units


in, cm
HeightUnk
Height not measured


1 => Checked
BldPrSys
Blood pressure systolic
60
300

BldPrDia
Blood pressure diastolic
30
160

BldPrUnk
Blood pressure not measured


1 => Checked
PEAbnormal
Were any clinically significant abnormalities found


Yes, No
PreExistMedCond
Does subject have any pre-existing medical conditions other than T1D


Yes, No
PtCurrMed
Is subject currently taking any medication


Yes, No
SHNumEverB
Estimated number of severe hypoglycemic events ever (as defined below)


0, 1, 2, 3, 4, 5-10, >10
SHMostRecentB
Estimate of when most severe hypoglycemic event (as defined below) occurred


< 3 months ago, 3-<6 months ago, 6-<12 months ago, 1-<5 years ago, 5-<10 years ago, >=10 years ago
SHLast12MonthsB
If <12 months ago, how many events occurred in the last 12 months


1, 2, 3, 4, 5-10, >10
SHSeizComaNumB
Estimated number of severe hypoglycemic events involving seizure/loss of consciousness ever


0, 1, 2, 3, 4, 5-10, >10
SHSeizComaLast12MonthsB
If 1 or more severe hypoglycemic events involving seizure/loss of consciousness, how many events occurred in the last 12 months


0, 1, 2, 3, 4, 5-10, >10
DKANumEverB
Estimated number of definite or probable DKA events ever (as defined below and including DKA at diagnosis)


0, 1, 2, 3, 4, 5-10, >10
DKAMostRecentB
If 1 or more definite or probable DKA events, estimate of when most recent definite or probable DKA event occurred


< 3 months ago, 3-<6 months ago, 6-<12 months ago, 1-<5 years ago, 5-<10 years ago, >=10 years ago
DKALast12MonthsB
If <12 months ago, how many events occurred in the last 12 months


1, 2, 3, 4, 5-10, >10


DiabSocioEcon -- Diabetes Socioeconomic Information form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



EducationLevel
Education Level


Less than 1st grade, 1st, 2nd, 3rd, or 4th grade, 5th or 6th grade, 7th or 8th grade, 9th grade, 10th grade, 11th grade, 12th grade - no diploma, High school graduate/diploma/GED, Some college but no degree, Associate Degree (AA), Bachelor's Degree (BS,BA,AB), Master's Degree (MA, MS, MSW, MBA, MPH), Professional Degree (MD, DDS, DVM, LLB, JD), Doctorate Degree (PhD, EdD), Unknown, Does not wish to provide
AnnualIncome
Annual Income


Less than $25,000, $25,000 to less than $35,000, $35,000 to less than $50,000, $50,000 to less than $75,000, $75,000 to less than $100,000, $100,000 to less than $200,000, $200,000 or more, Unknown, Does not wish to provide
InsPrivate
Private Insurance


1 => Checked
InsMedicare
Medicare Insurance


1 => Checked
InsMediGap
MediGap insurance


1 => Checked
InsMedicaid
Medicaid insurance


1 => Checked
InsSCHIP
SCHIP insurance


1 => Checked
InsMilitary
Military health care


1 => Checked
InsIndian
Indian health service plan


1 => Checked
InsState
State sponsered health plan


1 => Checked
InsOtherGov
Other Government health plan


1 => Checked
InsSingleService
Single service plan


1 => Checked
InsNoCoverage
No insurance coverage


1 => Checked
InsUnk
Insurance coverage unknown


1 => Checked
InsNoAns
Insurance coverage no answer


1 => Checked


DiabUnschedContact -- Diabetes unscheduled contact form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Unique Patient Identifier



ContType
Contact Type


Clinic visit, Phone call, Email, Text, Home visit, Video chat/Skype
ContPers
Contact was made with


Participant, Parent/Guardian, Spouse or significant other, Other
ContReasDevTrain
Reason for Contact: Participant needed additional device training


1 => Checked
ContReasProtTrain
Reason for Contact: Participant needed additional protocol/procedural training


1 => Checked
ContReasDiabMgmt
Reason for Contact: Participant had a question or problem with diabetes management


1 => Checked
ContReasAE
Reason for Contact: Participant had a potential adverse event


1 => Checked
ContReasDevIssue
Reason for Contact: Participant had a potential device deficiency/issue


1 => Checked
ContReasSupplies
Reason for Contact: Participant needed study supplies


1 => Checked
ContReasOth
Reason for Contact: Other


1 => Checked
ContPersPartic
Contact was made with: Participant


1 => Checked
ContPersParent
Contact was made with: Parent/Guardian


1 => Checked
ContPersSpouse
Contact was made with: Spouse or significant other


1 => Checked
ContPersOth
Contact was made with: Other


1 => Checked


FollowUpCTV -- List of Follow-Up Complete-the-Visit forms
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



CGMUploaded
Was the participant’s CGM data uploaded by the participant at home for clinician review of CGM data


Yes, No
CGMUploadedDt
If yes, enter the date the participant last uploaded data:



PumpSettingsChanged
Were any changes made to the participant’s insulin pump settings to optimize therapy


Yes, No
PumpSettingsChangedForSafety
If yes, were the changes made to address safety concerns


Yes, No
QuestionsAboutStudyDevices
Did the participant have any questions about using any of the study devices


Yes, No
DLCGM
Indicate which devices were downloaded: Study CGM receiver


1 => Checked
DLStudyPump
Indicate which devices were downloaded: Study Insulin Pump


1 => Checked
DLPersonalPump
Indicate which devices were downloaded: Personal Insulin Pump


1 => Checked
DLBGMeter
Indicate which devices were downloaded: Study blood glucose meter


1 => Checked
DLKetoneMeter
Indicate which devices were downloaded: Study blood ketone meter


1 => Checked
SuppliesAssigned
CGM, Blood Glucose Meter, Ketone Meter, and if applicable, Pump supplies were assigned to participant through inventory tracking during this visit


Yes, No
Weight
Weight
1


WeightUnits
Weight units


lbs, kg
Height
Height
1


HeightUnits
Height units


in, cm


Insulin -- List of all types of insulins for patient
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient ID



ParentInsulinListID
Insulin Name



InsRoute
Route


Pump, Injection, Inhaled
InsInjectionFreq
If injection or inhaled, what is the usual frequency of injections or inhaled per day


1, 2, 3, 4, 5, 6, 7, 8, 9, Unknown
InsTypeStart
Start Date of Insulin Type


In use at time of enrollment, Started after enrollment, Unknown
InsTypeStartDt
If started after enrollment, start date



InsTypeStartUnknown
If started after enrollment, start date: Unknown


1 => Checked
InsTypeStopDt
Stop Date of Insulin Type (if permanently discontinued during the study)



InsTypeStopUnknown
Stop Date of Insulin Type: Unknown


1 => Checked
InsTypeStartEstimate
If started after enrollment, start date: Estimated


1 => Checked
InsTypeStopEstimate
Stop Date of Insulin Type: Estimated


1 => Checked


InsulinPumpSettings -- Insulin Therapy form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient ID



NotApplicable
N/A, pump not being used


1 => Checked
InsTherapyDt
Effective date of insulin pump settings below



InsTherapyDtApprox
Approximate


1 => Checked
CurrTotInsDaily
Average Total Daily Insulin dose recorded on pump  from the preceding 7 days (IU/Day)



CurrTotInsDailyNA
Average total daily insulin not available


1 => Checked
TotBasalPreced7Days
Average daily basal dose recorded on pump from the preceding 7 days



TotBasalPreced7DaysNA
Average daily basal not available


1 => Checked
InsBasal0000
Insulin Basal Rates at 00:00
0.00
6.00

InsBasal0030
Insulin Basal Rates at 00:30
0.00
6.00

InsBasal0100
Insulin Basal Rates at 01:00
0.00
6.00

InsBasal0130
Insulin Basal Rates at 01:30
0.00
6.00

InsBasal0200
Insulin Basal Rates at 02:00
0.00
6.00

InsBasal0230
Insulin Basal Rates at 02:30
0.00
6.00

InsBasal0300
Insulin Basal Rates at 03:00
0.00
6.00

InsBasal0330
Insulin Basal Rates at 03:30
0.00
6.00

InsBasal0400
Insulin Basal Rates at 04:00
0.00
6.00

InsBasal0430
Insulin Basal Rates at 04:30
0.00
6.00

InsBasal0500
Insulin Basal Rates at 05:00
0.00
6.00

InsBasal0530
Insulin Basal Rates at 05:30
0.00
6.00

InsBasal0600
Insulin Basal Rates at 06:00
0.00
6.00

InsBasal0630
Insulin Basal Rates at 06:30
0.00
6.00

InsBasal0700
Insulin Basal Rates at 07:00
0.00
6.00

InsBasal0730
Insulin Basal Rates at 07:30
0.00
6.00

InsBasal0800
Insulin Basal Rates at 08:00
0.00
6.00

InsBasal0830
Insulin Basal Rates at 08:30
0.00
6.00

InsBasal0900
Insulin Basal Rates at 09:00
0.00
6.00

InsBasal0930
Insulin Basal Rates at 09:30
0.00
6.00

InsBasal1000
Insulin Basal Rates at 10:00
0.00
6.00

InsBasal1030
Insulin Basal Rates at 10:30
0.00
6.00

InsBasal1100
Insulin Basal Rates at 11:00
0.00
6.00

InsBasal1130
Insulin Basal Rates at 11:30
0.00
6.00

InsBasal1200
Insulin Basal Rates at 12:00
0.00
6.00

InsBasal1230
Insulin Basal Rates at 12:30
0.00
6.00

InsBasal1300
Insulin Basal Rates at 13:00
0.00
6.00

InsBasal1330
Insulin Basal Rates at 13:30
0.00
6.00

InsBasal1400
Insulin Basal Rates at 14:00
0.00
6.00

InsBasal1430
Insulin Basal Rates at 14:30
0.00
6.00

InsBasal1500
Insulin Basal Rates at 15:00
0.00
6.00

InsBasal1530
Insulin Basal Rates at 15:30
0.00
6.00

InsBasal1600
Insulin Basal Rates at 16:00
0.00
6.00

InsBasal1630
Insulin Basal Rates at 16:30
0.00
6.00

InsBasal1700
Insulin Basal Rates at 17:00
0.00
6.00

InsBasal1730
Insulin Basal Rates at 17:30
0.00
6.00

InsBasal1800
Insulin Basal Rates at 18:00
0.00
6.00

InsBasal1830
Insulin Basal Rates at 18:30
0.00
6.00

InsBasal1900
Insulin Basal Rates at 19:00
0.00
6.00

InsBasal1930
Insulin Basal Rates at 19:30
0.00
6.00

InsBasal2000
Insulin Basal Rates at 20:00
0.00
6.00

InsBasal2030
Insulin Basal Rates at 20:30
0.00
6.00

InsBasal2100
Insulin Basal Rates at 21:00
0.00
6.00

InsBasal2130
Insulin Basal Rates at 21:30
0.00
6.00

InsBasal2200
Insulin Basal Rates at 22:00
0.00
6.00

InsBasal2230
Insulin Basal Rates at 22:30
0.00
6.00

InsBasal2300
Insulin Basal Rates at 23:00
0.00
6.00

InsBasal2330
Insulin Basal Rates at 23:30
0.00
6.00

InsBolusStart1Hr
Insulin Bolus Hour Start of Time Range - Line 1


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart1Min
Insulin Bolus Minute Start of Time Range - Line 1


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd1Hr
Insulin Bolus Hour End of Time Range - Line 1


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd1Min
Insulin Bolus Minute End of Time Range - Line 1


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio1
CHO Ratio (1:x gCHO)- Line 1
0
99

CorrFactor1
Correction Factor (1:x mg/dl) - Line 1
10.0
400.0

InsBolusStart2Hr
Insulin Bolus Hour Start of Time Range - Line 2


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart2Min
Insulin Bolus Minute Start of Time Range - Line 2


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd2Hr
Insulin Bolus Hour End of Time Range - Line 2


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd2Min
Insulin Bolus Minute End of Time Range - Line 2


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio2
CHO Ratio (1:x gCHO)- Line 2
0
99

CorrFactor2
Correction Factor (1:x mg/dl) - Line 2
10.0
400.0

InsBolusStart3Hr
Insulin Bolus Hour Start of Time Range - Line 3


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart3Min
Insulin Bolus Minute Start of Time Range - Line 3


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd3Hr
Insulin Bolus Hour End of Time Range - Line 3


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd3Min
Insulin Bolus Minute End of Time Range - Line 3


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio3
CHO Ratio (1:x gCHO)- Line 3
0
99

CorrFactor3
Correction Factor (1:x mg/dl) - Line 3
10.0
400.0

InsBolusStart4Hr
Insulin Bolus Hour Start of Time Range - Line 4


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart4Min
Insulin Bolus Minute Start of Time Range - Line 4


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd4Hr
Insulin Bolus Hour End of Time Range - Line 4


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd4Min
Insulin Bolus Minute End of Time Range - Line 4


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio4
CHO Ratio (1:x gCHO)- Line 4
0
99

CorrFactor4
Correction Factor (1:x mg/dl) - Line 4
10.0
400.0

InsBolusStart5Hr
Insulin Bolus Hour Start of Time Range - Line 5


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart5Min
Insulin Bolus Minute Start of Time Range - Line 5


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd5Hr
Insulin Bolus Hour End of Time Range - Line 5


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd5Min
Insulin Bolus Minute End of Time Range - Line 5


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio5
CHO Ratio (1:x gCHO)- Line 5
0
99

CorrFactor5
Correction Factor (1:x mg/dl) - Line 5
10.0
400.0

InsBolusStart6Hr
Insulin Bolus Hour Start of Time Range - Line 6


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart6Min
Insulin Bolus Minute Start of Time Range - Line 6


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd6Hr
Insulin Bolus Hour End of Time Range - Line 6


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd6Min
Insulin Bolus Minute End of Time Range - Line 6


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio6
CHO Ratio (1:x gCHO)- Line 6
0
99

CorrFactor6
Correction Factor (1:x mg/dl) - Line 6
10.0
400.0

InsBolusStart7Hr
Insulin Bolus Hour Start of Time Range - Line 7


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart7Min
Insulin Bolus Minute Start of Time Range - Line 7


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd7Hr
Insulin Bolus Hour End of Time Range - Line 7


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd7Min
Insulin Bolus Minute End of Time Range - Line 7


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio7
CHO Ratio (1:x gCHO)- Line 7
0
99

CorrFactor7
Correction Factor (1:x mg/dl) - Line 7
10.0
400.0

InsBolusStart8Hr
Insulin Bolus Hour Start of Time Range - Line 8


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart8Min
Insulin Bolus Minute Start of Time Range - Line 8


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd8Hr
Insulin Bolus Hour End of Time Range - Line 8


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd8Min
Insulin Bolus Minute End of Time Range - Line 8


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio8
CHO Ratio (1:x gCHO)- Line 8
0
99

CorrFactor8
Correction Factor (1:x mg/dl) - Line 8
10.0
400.0

InsBolusStart9Hr
Insulin Bolus Hour Start of Time Range - Line 9


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart9Min
Insulin Bolus Minute Start of Time Range - Line 9


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd9Hr
Insulin Bolus Hour End of Time Range - Line 9


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd9Min
Insulin Bolus Minute End of Time Range - Line 9


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio9
CHO Ratio (1:x gCHO)- Line 9
0
99

CorrFactor9
Correction Factor (1:x mg/dl) - Line 9
10.0
400.0

InsBolusStart10Hr
Insulin Bolus Hour Start of Time Range - Line 10


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusStart10Min
Insulin Bolus Minute Start of Time Range - Line 10


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
InsBolusEnd10Hr
Insulin Bolus Hour End of Time Range - Line 10


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
InsBolusEnd10Min
Insulin Bolus Minute End of Time Range - Line 10


0 => 00, 1 => 01, 2 => 02, 3 => 03, 4 => 04, 5 => 05, 6 => 06, 7 => 07, 8 => 08, 9 => 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
CHORatio10
CHO Ratio (1:x gCHO)- Line 10
0
99

CorrFactor10
Correction Factor (1:x mg/dl) - Line 10
10.0
400.0

SettingsUnchanged
Settings unchanged from prior form submission


1 => Checked


MedicalCondition -- Medical Condition Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Subject ID



ParentMedCondListID
Condition



MedCondPreStart
Medical condition present prior to study enrollment


Yes, No
MedCondPreStartCat
If present prior to enrollment, approximate duration (prior to enrollment)


<=30 days, >30 days to < 3 months, 3 months to < 6 months, 6 months to < 1 year, 1 year to < 5 years, 5 years to < 10 years, >=10 years, Unknown
MedCondPreStartTreat
If present prior to enrollment, was the medical condition treated with medication


Current, Past, Never
MedCondDiagDt
If condition started after enrollment, date of diagnosis



MedCondDiagDtApprox
Date of diagnosis is approximate


1 => Checked
MedCondDiagDtUnk
Date of diagnosis is unknown


1 => Checked
MedCondTrt
If condition started after enrollment, treatment of the medical condition


None, Medication, Surgery, Medication and Surgery, Dietary Management, Other
MedCondResDt
 Recovery Date



MedCondResDtApprox
Recovery date of medical condition is approximate


1 => Checked
MedCondStatus
Medical condition status


Ongoing (further improvement / worsening possible) => Ongoing (further improvement or worsening possible), Ongoing, medically stable => Ongoing, medically stable (further change not expected), Complete Recovery, Recovered with Sequelae
MedCondDiagMonth
If condition started after enrollment, month of diagnosis


1 => Jan, 2 => Feb, 3 => Mar, 4 => Apr, 5 => May, 6 => Jun, 7 => Jul, 8 => Aug, 9 => Sep, 10 => Oct, 11 => Nov, 12 => Dec
MedCondDiagYear
If condition started after enrollment, year of diagnosis



MedCondCurrTreatMed
Current treatment with medications


Yes, No


Medication -- Medications Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Subject ID



MedDose
Dose per administration



MedUnit
Dose per administration unit


aerosol, ampules, capsules, cream, dl => dl-deciliter, elixir, g => g-gram, gal => gal-gallon, gtt-drops, IU => IU-International unit, kg => kg-kilogram, L => L-liter, lbs => lbs-pounds, M => M-molar, mcg => mcg-microgram, meq => meq-milliequivalent, mg => mg-milligram, mL => mL-milliliter, Ointment, pills, pt => pt-pint, puff, ounces, qt => qt-quart, tablet, tbs => tbs-tablespoon, tsp => tsp-teaspoon, ul => ul-microliter, units, vials
MedDoseUnk
Dose per administration is unknown


1 => Checked
MedRoute
Route


PO => P.O.-by mouth, SC => S.C.-subcutaneous, Topical - ocular => Topical - ocular (not drops), Gtt => Gtt-drops, Intravitreal, IV => I.V.-intravenous, IM => I.M.-intramuscular, Oral Inhalation, Topical - skin, Nasal, Other, Topical, ID => I.D.-intradermal, PR => P.R.-by rectum, Vaginal, Transurethral, Sublingual, Peribulbar, Intra-articular => Intra-articular injection, Retrobulbar, Transdermal, Subconjunctival, Subtenons, Intrauterine, Epidural, Gastrostomy tube, Intracavernous
MedLocSide
If treatment is for eye or ear, which was treated with medication


Right, Left, Both
MedFreqType
Frequency


Fixed Regimen, As Needed, One Time Treatment
MedFreqNum
Frequency number


1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
MedFreqPer
Frequency period


Day, Week, Month, Year
MedFreqUnk
Medication frequency is unknown


1 => Checked
MedInd
Indication


Prior to enrollment => Medical condition prior to enrollment, New => New medical condition/adverse event, Prevention
ParentLoginIDMedCondition
Medical Condition



ParentLoginIDAdvEvent
Adverse Event



MedStartTrtCat
Start date of treatment category


On treatment at time of enrollment, Treatment started after enrollment
MedStartDt
If treatment started after enrollment/randomization, Medication Start Date



MedStartDtApprox
Medication Start Date is approximate


1 => Checked
MedStopDt
Medication Stop Date



MedStopDtApprox
Medication Stop Date is approximate


1 => Checked
MedOngoing
Medication is ongoing


1 => Checked
MedStartPreEnrRange
If on treatment at time of enrollment, Medication Start Range


<=30 days, >30 days to < 3 months, 3 months to < 6 months, 6 months to < 1 year, 1 year to < 5 years, 5 years to < 10 years, >=10 years, Unknown
MedStartMonth
If treatment started after enrollment, Medication Start Month


1 => Jan, 2 => Feb, 3 => Mar, 4 => Apr, 5 => May, 6 => Jun, 7 => Jul, 8 => Aug, 9 => Sep, 10 => Oct, 11 => Nov, 12 => Dec
MedStartYear
If treatment started after enrollment, Medication Start Year



MedStopMonth
Medication stop Month


1 => Jan, 2 => Feb, 3 => Mar, 4 => Apr, 5 => May, 6 => Jun, 7 => Jul, 8 => Aug, 9 => Sep, 10 => Oct, 11 => Nov, 12 => Dec
MedStopYear
Medication stop Year



MedStopDtUnk
Medication stop date unknown


1 => Checked
ParentLogInIDPreExisting
PreExisting Condition



MedStartDtUnk
Medication start date unknown


1 => Checked
MedCondNotReqd
Condition not required to be reported on medical condition form


1 => Checked
PreExistCondNotReqd
Condition not required to be reported on pre-existing condition form


1 => Checked
ParentRxNormDrugListID
Medication



AdvEventNotReqd
Condition not required to be reported on adverse event form


1 => Checked
ParentLoginIDMedCondition2
Medical Condition



ParentLogInIDPreExisting2
PreExisting Condition



ParentLoginIDAdvEvent2
Adverse Event





PtFinalStat -- Final Status Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



PreRandPtFinalStatReas
Reason participant's participation in the Study has ended (Pre-rand reason)


ID obtained in error => ID obtained in error - No study data collected, Participant does not meet screening eligibility => Participant does not meet all screening eligibility criteria - detail in COMMENTS, Lost to follow up prior to randomization => Lost to follow up prior to randomization - detail efforts to contact participant in COMMENTS, Requests to withdraw not in writing => Participant/Parent requests to withdraw - did not withdraw consent in writing, Requests to withdraw in writing => Participant/Parent requests to withdraw - formally withdrew consent in writing, Site withdraws participant => Site withdraws participant - indicate reason in COMMENTS, Death
PostRandPtFinalStatReas
Reason participant's participation in the Study has ended (Post-rand reason)


Requests to withdraw not in writing => Participant/Parent requests to withdraw - did not withdraw consent in writing, Requests to withdraw in writing => Participant/Parent requests to withdraw - formally withdrew consent in writing, Lost to follow up => Lost to follow up - detail efforts to contact participant in COMMENTS, Site withdraws participant => Site withdraws participant - indicate reason in COMMENTS, Death
PtFinalStatReas
Reason participant's participation in the Study has ended


ID obtained in error => ID obtained in error - No study data collected, Participant does not meet screening eligibility => Participant does not meet all screening eligibility criteria - detail in COMMENTS, Requests to withdraw not in writing => Participant/Parent requests to withdraw - did not withdraw consent in writing, Requests to withdraw in writing => Participant/Parent requests to withdraw - formally withdrew consent in writing, Lost to follow up => Lost to follow up - detail efforts to contact participant in COMMENTS, Site withdraws participant => Site withdraws participant - indicate reason in COMMENTS, Death
PtWithdrawAE
Reason for participant/parent withdrawal: Adverse event


1 => Checked
PtWithdrawChgDr
Reason for participant/parent withdrawal: Changed doctor


1 => Checked
PtWithdrawNoStdyTrt
Reason for participant/parent withdrawal: Does not want study treatment


1 => Checked
PtWithdrawFin
Reason for participant/parent withdrawal: Finances


1 => Checked
PtWithdrawChgIns
Reason for participant/parent withdrawal: Changed insurance


1 => Checked
PtWithdrawMoved
Reason for participant/parent withdrawal: Moved


1 => Checked
PtWithdrawOthTrtReq
Reason for participant/parent withdrawal: Other treatment requested


1 => Checked
PtWithdrawPoorHealth
Reason for participant/parent withdrawal: Poor health


1 => Checked
PtWithdrawPoorOut
Reason for participant/parent withdrawal: Poor outcome


1 => Checked
PtWithdrawSched
Reason for participant/parent withdrawal: Scheduling/availability issues


1 => Checked
PtWithdrawTravDiff
Reason for participant/parent withdrawal: Travel difficulty


1 => Checked
PtWithdrawVisLen
Reason for participant/parent withdrawal: Visit too lengthy


1 => Checked
PtWithdrawUnk
Reason for participant/parent withdrawal: Unknown


1 => Checked
DeathDt
Date of Death





PtRoster -- List of study participants
Name
Description
Min
Max
Possible_Values
PtID
Participant ID



PtID
Patient Identifier



SiteID
Site Identifier



EnrollDt
Date of enrollment



RandDt
Date of randomization



PtStatus
Patient Status


Active, Completed, Dropped
TrtGroup
Treatment group randomization assignment


CLC, SAP


QCTest -- Device quality control testing for the Blood Glucose Meter and the Blood Ketone Meter
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient ID



QCMeter
Was QC testing successful with the study blood glucose meter using two different concentrations of control solution?


Yes, No
QCKetone
Was QC testing successful with the study blood ketone meter using two different concentrations of control solution?


Yes, No


RandomizationCTV -- Randomization Complete the Visit Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



ProceduresComplete
Are all required procedures for randomization to occur complete


Yes, No


RunInReview -- List of Run-In Review data
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



CGMUsed11Days
Was the minimum required 11 days of sensor use out of 14 days of home use obtained, based on CGM data download and review


Yes, No
PumpUsedDaily
Was the study insulin pump used each day based on pump data download and review


Yes, No
PumpUsedDailyExplan
If No, what is the reason


PersonalPump => Participant using personal insulin pump, CGMNaive and MDI => Participant is MDI and CGM-naïve and not yet assigned study pump, Other
NeedMoreRunIn
Does the participant require another two-week Run-in period


Yes, No
NeedMoreRunInCGM
Insufficient study CGM use


1 => Checked
NeedMoreRunInPump
Insufficient study pump use


1 => Checked
PumpSettingsChanged
Were changes made to pump settings


Yes, No, N/A => N/A, pump not used during this run-in review period
SkinReaction
Did the participant experience a skin reaction in the area where the CGM sensor was worn


Yes, No
SkinReactionRedness
Skin reaction: check all that apply: Redness


1 => Checked
SkinReactionSwelling
Skin reaction: check all that apply: Redness


1 => Checked
SkinReactionPain
Skin reaction: check all that apply: Pain


1 => Checked
SkinReactionBruising
Skin reaction: check all that apply: Bruising


1 => Checked


ScreeningCTV -- Screening Complete the Visit Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



DexcomUse
Does the participant currently use a Dexcom CGM with use on at least 11 of the prior 14 days that satisfies the protocol’s CGM requirement for skipping run-in


Yes, No


Training -- List of Study Device Training forms
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Participant ID



PumpTrained
Was study pump training and corresponding checklist completed at this visit


Yes, No
PumpTrainedPumpType
If “Yes,” indicate pump type/checklist completed and trainer details


t:slim X2 with Control-IQ, t:slim X2 with no automated insulin delivery
PumpTrainedReason
If “No”, please indicate why not


PersonalPump => Participant is using a personal pump, PriorVisit => Study pump training was delivered and documented at a prior visit, CGMNaive and MDI => Participant is CGM naive MDI user beginning run-in, None => None of the reasons listed above
CGMTrained
Was the study CGM training and corresponding checklist completed at this visit


Yes, No-TrainedPriorVisit => No, study CGM training was delivered and documented at a prior visit, No-ReasonNotListed => No, for a reason not listed
SecondVisit
Did the participant require a second visit for completion of study pump and Control-IQ system training


Yes, No, N/A => N/A, participant not using t:slim X2 with Control-IQ
SecondVisitDt
Training completion date





VisitInfo -- Visit Information Form
Name
Description
Min
Max
Possible_Values
ParentLoginVisitID
If this column is populated, the CRF is part of a visit, and can be joined on that column to the VisitInfo table to get information on the visit.



RecID
Unique record ID in table



PtID
Patient Identifier



Visit
Visit



VisitDt
Visit Date



OutOfWin
Visit was completed out of window


1 => Checked
OutOfWinReason
Reason visit was completed out of window


Bad weather, Travel difficulty, Financial issue, Poor health, Personal issue, Work issue, Subject on vacation, Visits too lengthy, Investigator away, Clinic appointment not available, Site forgot to schedule, Difficulty contacting subject, Poor outcome, Good outcome, Adverse event, Unknown, Other
VisitMiss
Visit was missed


1 => Checked
VisitMissReason
Reason visit was missed


Bad weather, Travel difficulty, Financial issue, Poor health, Personal issue, Work issue, Subject on vacation, Visits too lengthy, Investigator away, Clinic appointment not available, Site forgot to schedule, Difficulty contacting subject, Poor outcome, Good outcome, Adverse event, Unknown, Other


DexcomClarityCGM -- CGM readings from Dexcom Clarity files
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



DexcomIndex
Index column from Dexcom Clarity file



PtID
Participant ID



DataDtTm*
Time of event



EventSubType
Event subtype



CGM
Glucose value (mg/dL) from Clarity file



TransmitterTime
Transmitter time



DataDtTm_adj
Local time of event, including adjustments made at Jaeb




* Note: participant ID 68 has approximately 2 weeks of pre-admission baseline Dexcom Clarity data with deidentified dates between 1999-04-20 and 1999-05-03. These deidentified dates are not accurate with respect to the deidentified enrollment date (2001-02-05) of the participant; the correct relative dates are not known, but the data should be interpreted as representing baseline CGM control in the approximate two-week period prior to the participant’s enrollment.   
DexcomClarityMeter -- Calibration events recorded in Dexcom Clarity files
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



DexcomIndex
Index from Clarity file



PtID
Participant ID



DataDtTm*
Time of calibration event



EventSubType
Event subtype



Meter
Glucose Value (mg/dL)



DataDtTm_adj
Local time of event, including adjustments made at Jaeb




* Note: participant ID 68 has approximately 2 weeks of pre-admission baseline Dexcom Clarity data with deidentified dates between 1999-04-20 and 1999-05-03. These deidentified dates are not accurate with respect to the deidentified enrollment date (2001-02-05) of the participant; the correct relative dates are not known, but the data should be interpreted as representing baseline CGM control in the approximate two-week period prior to the participant’s enrollment.   
cgm – dataset that aggregates CGM data from multiple sources and is ready for analyses
Order
Variable
DataType
Description
1
PtID
Character/$30
De-identified PtID
2
Period
Character/$40
Gives the Study Phase: Baseline vs. Post-Randomization
3
DataDtTm*
Numeric/datetime16.0
De-identified CGM reading DtTm
4
CGM
Numeric/7.0
CGM reading

* Note: participant ID 68 has approximately 2 weeks of pre-admission baseline Dexcom Clarity CGM data with deidentified dates between 1999-04-20 and 1999-05-03. These deidentified dates are not accurate with respect to the deidentified enrollment date (2001-02-05) of the participant; the correct relative dates are not known, but the data should be interpreted as representing baseline CGM control in the approximate two-week period prior to the participant’s enrollment.   
gluIndices – glycemic indices calculated under several assumption; it uses the previous cgm dataset as input
Order
Variable
DataType
Description
1
PtID
Character/$30
De-identified PtID
2
analysis
Character/$40
Analysis: 24hr, day/night (use with daytime), first 3 months, exclude first 2 weeks, safety
3
dayTime
Numeric/8.0
Indicator for daytime (6:00AM to Midnight) – to be used for analysis=”day/night”
4
period
Character/$40
Gives the Study Phase: Baseline vs. Post-Randomization
5
gluHours
Numeric/8.1
Hours of glucose readings
6
gluBelow54
Numeric/percent10.3
% below 54 mg/dL
7
gluBelow60
Numeric/percent10.3
% below 60 mg/dL
8
gluBelow70
Numeric/percent10.3
% below 70 mg/dL
9
gluLBGI
Numeric/8.2
Low BG index
10
gluHypoRate
Numeric/8.2
Hypoglycemic Event Rate per Week
11
gluInRange
Numeric/percent10.1
% in range 70-180 mg/dL
12
gluInRange140
Numeric/percent10.1
% in range 70-140 mg/dL
13
gluCV
Numeric/percent10.1
Coefficient of variation=gluSD/gluMean
14
gluSD
Numeric/8.1
Standard deviation
15
gluMean
Numeric/8.1
Mean glucose
16
gluAbove180
Numeric/percent10.1
% above 180 mg/dL
17
gluAbove250
Numeric/percent10.1
% above 250 mg/dL
18
gluAbove300
Numeric/percent10.1
% above 300 mg/dL
19
gluHBGI
Numeric/8.2
High BG index


DiasendCGM -- Readings from CGM tab of Diasend file
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



PtID
Participant ID



DataDtTm
Time of reading



DataDtTm_adjusted
Local time of event, including adjustments made at Jaeb



CGM
Sensor glucose reading (mg/dL)



IsCalibration
Is the reading a calibration


Yes, No


OtherCGM -- List of CGM readings from other miscellaneous sources
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



PtID
Participant ID



DataDtTm
Datetime of CGM Reading



DataDtTm_adjusted
Local time of event, including adjustments made at Jaeb



CGM
CGM (mg/dL)



IsCalibration



Yes, No



SampleResults – List of lab results
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



PtID
Participant ID



Visit
Visit



AnalysisDt
Date analysis run by the lab



ResultName
Name of the result


GLYHB => Glycated hemoglobin (HbA1c), CPEP => C-peptide (nonfasting), GLU => blood glucose value associated with C-peptide measurement
Value
Result value



Units
Units of measurement



CollectionDt
Date blood sample was collected from participant




clarkeHypoAwarness - Clark Hypoglycemia Awareness Questionnaire administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
diabetesDistressAdult – Diabetes Distress Questionnaire (adult version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
diabetesDistressParent – Diabetes Distress Questionnaire (teen’s parent version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
diabetesPersonality – Diabetes Specific Personality Questionnaire administered at Baseline. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
hyperglycemiaAvoidance – Hyperglycemia Avoidance Scale administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
hypoglycemiaConfidence – Hypoglycemia Confidence Scale administered at Baseline, 13-week, and 26-week
Authors: Polonsky WH, Fisher L, Hessler D, Edelman SV
https://behavioraldiabetes.org/scales-and-measures/#1486573022939-e501f79a-cec3
Order
Variable
DataType
Description
1
PtID
Character/$30
De-identified PtID
2
hypoConfQDt
Numeric/date7.0
De-identified Survey DtTm
3
when_exrcse
Character/$100
When you are exercising?
4
when_sleep
Character/$100
When you are sleeping?
5
when_drive
Character/$100
When you are driving?
6
when_soc_sit
Character/$100
When you are in social situations?
7
when_alne
Character/$100
When you are alone?
8
avd_prob_hypo
Character/$100
Avoid serious problems due to hypoglycemia?
9
catch_respnd
Character/$100
Catch and respond to hypoglycemia before your blood sugars get too low?
10
cont_despite_hypo
Character/$100
Continue to do the things you really want to do in your life, despite the risks of hypoglycemia?
11
confdt_spouse
Character/$100
If you have a spouse or partner: What is your best guess about how confident your spouse or partner feels about your ability to avoid serious problems due to hypoglycemia?
12
hcs_unanswred
Character/$100
You have unanswered questions. If you have accidentally left a question unanswered, please go back at this moment and answer those questions.





hypoglycemiaFearAdult – Hypoglycemia Fear Survey (adult version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
hypoglycemiaFearTeen – Hypoglycemia Fear Survey (teen version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
hypoglycemiaFearTeenParent – Hypoglycemia Fear Survey (teen’s parent version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
inspireAdult – INSPIRE Survey (adult version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
inspireTeen – INSPIRE Survey (teen version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
inspireTeenParent – INSPIRE Survey (teen’s parent version) administered at Baseline, 13-week, and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
systemUsability – System Usability Scale administered at 13-week and 26-week. 
https://hell.meiert.org/core/pdf/sus.pdf
Order
Variable
DataType
Description
1
PtID
Character/$30
De-identified PtID
2
sysUsabQDt
Numeric/date7.0
De-identified Survey DtTm
3
use_frequent
Character/$100
1. I think that I would like to use this system frequently
4
unness_cmplx
Character/$100
2. I found the system unnecessarily complex
5
easy_use
Character/$100
3. I thought the system was easy to use
6
need_support
Character/$100
4. I think that I would need the support of a technical person to be able to use this system
7
well_integrated
Character/$100
5. I found the various functions in this system were well integrated
8
much_inconsist
Character/$100
6. I thought there was too much inconsistency in this system
9
learn_quick
Character/$100
7. I would imagine that most people would learn to use this system very quickly
10
vry_cumbrsme
Character/$100
8. I found the system very cumbersome to use
11
vry_confident
Character/$100
9. I felt very confident using the system
12
learn_lot
Character/$100
10. I needed to learn a lot of things before I could get going with this system
13
susTotScore
Numeric/8.1
Total Score [(sum of odd items - 5 + 25 - sum of even items) multiplied by 2.5]
14
susBenefitAvgScore
Numeric/8.1
Average Benefit Score (Average of odd items: E=5, ..., A=1)
15
susBarrierAvgScore
Numeric/8.1
Average Barrier Score (Average of reverse-scored even items: E=5, ..., A=1)


technologyAcceptance – Technology Acceptance Survey administered at 13-week and 26-week. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.
technologyExpectations – Technology Expectations Survey administered at Baseline. 
Due to copyright, no further details are provided. Please inquire with the Jaeb Center for Health Research for additional information.

RocheMeter -- List of BG meter readings from Roche file
Name
Description
Min
Max
Possible_Values
RecID
Unique record ID in table



PtID
Participant ID



DataDtTm
Combination of date and time columns from Roche file



BG
bG (mg/dL)



Carbs
Carbohydrates (g)



IsQCTest
Whether a reading is a QC Test: Not from Review Table; calculated from EventsSystemDefined



SystemDefinedEvents
System-defined events



UserDefinedEvents
User-defined events



Flags
Flags



IsQCtestJaeb
Whether a reading is test(0) or control(1) -- this column may contain adjustments made at Jaeb



DataDtTm_adjusted
Local time of event, including adjustments made at Jaeb





Pump_BasalRateChange-- One record per basal rate change due to pumping events.
columnName
Description
Range
Units
PossibleValues
BasalRate
The new basal rate
0-25
units/hour

RecID
Unique record ID in table



PtID
Participant ID



DataDtTm
Local time of event



DataDtTm_adjusted
Local time of event -- Including adjustments made at Jaeb






Pump_BolusDelivered -- One record per bolus delivered by pump.
columnName
Description
Range
Units
PossibleValues
BolusAmount
Size of bolus delivered

Units of insulin

BolusType
Standard or Extended



RecID
Unique record ID in table – this data set comes from two tables, so there will be duplicate values in this column



PtID
Participant ID



DataDtTm
Local time of event



DataDtTm_adjusted
Local time of event -- Including adjustments made at Jaeb





Pump_CGMGlucoseValue-- One record per CGM glucose value recorded on the pump.
columnName
Description
Range
Units
PossibleValues
HighLowIndicator
 0 =CGMValue contains the glucose reading|1 =The glucose reading is high~ CGMValue set to 0|2 =The glucose reading is low~ CGMValue set to 0
0-5
Enum

CGMValue
40-400 are legitimate glucose values. 0 is used when the reading is high or low. Use of this field depends on the glucoseValueStatus field.
40-400
mg/dL

RecID
Unique record ID in table



PtID
Participant ID



DataDtTm
Local time of event



DataDtTm_adjusted
Local time of event -- Including adjustments made at Jaeb





