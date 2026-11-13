# 의료 기초 데이터 분석 미니 프로젝트
# Numpy + Pandas + Matplotlib

#====================================
# [0] 라이브러리  import, 출력 옵션(선택)
#====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#출력 옵션
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 200)

#====================================
# [1] 데이터 준비
#====================================

data = {
    '측정일': [
        '2026-02-01','2026-02-01','2026-02-02',
        '2026-02-02','2026-02-03','2026-02-03',
        '2026-02-04','2026-02-04','2026-02-05',
        '2026-02-05','2026-02-06','2026-02-06',
        '2026-02-07','2026-02-07','2026-02-08'
    ],
    '환자ID': [
        'P001','P002','P003','P004','P005',
        'P006','P007','P008','P009','P010',
        'P011','P012','P013','P014','P015'
    ],
    '성별': [
        'F','M','M','F','F','M','M','F',
        'F','M','F','M','M','F','F'
    ],
    '나이': [
        25,31,44,np.nan,52,39,28,33,
        47,61,29,42,np.nan,36,55
    ],
    '키': [
        '162cm','175cm','168cm','160cm','158cm',
        '180cm','172cm','165cm','170cm','177cm',
        '159cm','174cm','169cm','163cm','157cm'
    ],
    '몸무게': [
        '54kg','78kg','70kg','52kg','60kg',
        '90kg','68kg','58kg','73kg','82kg',
        '50kg','76kg','71kg','57kg','66kg'
    ],
    'SBP(수축기)': [
        108,135,142,118,150,130,125,112,
        140,160,105,138,145,120,155
    ],
    'DBP(이완기)': [
        70,88,92,75,95,85,80,72,
        90,100,68,89,96,78,97
    ],
    '심박수': [
        72,84,90,68,95,78,70,75,
        88,92,66,80,86,73,89
    ],
    '공복혈당': [
        92,110,125,98,130,np.nan,105,90,
        118,140,88,112,128,100,np.nan
    ],
    '흡연': [
        'N','Y','Y','N','N','Y','N','N',
        'Y','Y','N','Y','Y','N','N'
    ],
    '운동빈도(주)': [
        3,1,0,2,np.nan,1,4,2,
        1,0,5,1,0,3,np.nan
    ],
}

df = pd.DataFrame(data)

#====================================
# [2] 데이터 탐색
#====================================


print("===================================")
print("[2] 데이터 탐색")
print("====================================")

print("\n[head()]")
print(df.head())

print("\n[info()] / 데이터 구조 확인")
print(df.info())

print("\n[describe] / 숫자 칼럼 요약")
num_cols = ['나이', 'SBP(수축기)', 'DBP(이완기)', '심박수', '공복혈당', '운동빈도(주)']
print(df[num_cols].describe)

#====================================
# [3] 데이터 정제 및 변환
#====================================
print("===================================")
print("[3] 데이터 정제 및 변환")
print("====================================")

#3-1) 타압 변환: 날짜 문자열 -> datetime
print(df['측정일'].dtype)  #변환 전
df['측정일'] = pd.to_datetime(df['측정일'])
print(df['측정일'].dtype)  #변환 후

#3-2) 단위 제거: '170cm' -> 170, '70kg' -> 70
# str.replace() : 문자열 치환(여기서는 단위 제거)
# astype(int) : 숫자형으로 변환
print("3-2) 단위 제거: '170cm' -> 170, '70kg' -> 70")
print(df['키'].dtype)
print(df['몸무게'].dtype)

df['키_cm'] = df['키'].str.replace('cm', '', regex=False).astype(int)
df['몸무게_kg'] = df['몸무게'].str.replace('kg', '', regex=False).astype(int)

print(df['키_cm'].dtype)
print(df['몸무게_kg'].dtype)
# 원본(키, 몸무게) 칼럼은 '문자열'이라 분석에 불리하므로
# 보통 숫자형 칼럼(키_cm, 몸무게_kg)만 쓰는 편


#3-3) 결측치 확인
print("3-3) 결측치 확인")
print("[결측치 개수 확인]")
print(df.isnull().sum())

#3-4) 결측치 처리(예시)
'''
결측치 처리 방식은 정답이 1개가 아니라, 상황에 따라 다름
 - 나이 : 평균으로 채움
 - 공복혈당 : 중앙값으로 채움 (극단값 영향을 줄이고 싶을 때)
 - 운동빈도 : 0으로 채움(운동 기록이 없으면 0으로 가정하는 경우)
'''
df['나이'] = df['나이'].fillna(df['나이'].mean())
df['공복혈당'] = df['공복혈당'].fillna(df['공복혈당'].median()) #중앙값
df['운동빈도(주)'] = df['운동빈도(주)'].fillna(0)

print('[결측치 처리 후 개수 확인]')
print(df.isnull().sum())

#3-5) 파생 컬럼 생성(변환)
# BMI = 몸무게(kg) / 키(m)^2
df['키_m'] = df['키_cm'] / 100
df['BMI'] = df['몸무게_kg'] / (df['키_m']**2)

# 데이터 확인
print("==cm -> m 변환 ==")
print(df[['키_cm', '키_m']].head())
print("== BMI ==")
print(df['BMI'].head())

print("== 맥박압(PP) ==")
# 맥박압(PP) = SBP - DBP (혈관 탄성/압력 차이를 보는 간단 지표)
df['맥박압_PP'] = df['SBP(수축기)'] - df['DBP(이완기)']
print(df['맥박압_PP'].head())

print("== 평균동맥압(MAP) ==")
# 평균동맥압(MAP) = (SBP + 2*DBP) / 3 (기초적인 참고 지표)
df['평균동맥압_MAP'] = (df['SBP(수축기)'] + 2 * df['DBP(이완기)']) / 3
print(df['평균동맥압_MAP'].head())

#3-6) 범주화(카테고리 만들기) : 혈압 단계 분류
'''
기준(간단 버전)
 - 정상 : SBP < 120 그리고 DBP < 80
 - 주의(상승) : SBP 120~129 그리고 DBP < 80
 - 고혈압1기 : SBP 130~139 또는 DBP 80~89
 - 고혈압2기 : SBP >=140 또는 DBP >= 90
'''

def bp_category(sbp, dbp):
    if(sbp >= 140) or (dbp >= 90):
        return "고혈압2기"
    elif(130 <= sbp <= 139) or (80 <= dbp < 89):
        return "고혈압1기"
    # (120 <= sbp) and (sbp <=129)
    elif(120 <= sbp <= 129) and (dbp < 80):
        return "주의(상승)"
    elif(sbp < 120) and (dbp < 80):
        return "정상"
    else:
        return "기타"

'''
# 일반 함수
def test(row):
    return 'hi'

# lambda : 이름 없이 간단하게 작성하는 함수(익명 함수)
# lambda 매개변수: 반환할_표현식
lambda row: 'hi'
'''

#apply() :  행 단위로 함수를 적용할 때 자주 사용
'''
def classify_row(row):
    sbp = row['SBP(수축기)'],
    dbp = row['DBP(이완기)']

    return bp_category(sbp, dbp)

df['혈압분류'] = df.apply(classify_row, axis=1)
'''
df['혈압분류'] = df.apply(
    lambda row: bp_category(
        row['SBP(수축기)'], 
        row['DBP(이완기)']
        ), axis=1
    )

print("[혈압 분류 결과]")
print(df[['SBP(수축기)', 'DBP(이완기)', '혈압분류']].head())

#3-6) 범주화 : BMI 단계 분류(간단 예시)
'''
BMI 분류
 - 저체중: < 18.5
 - 정상 : 18.5 ~ 22.9
 - 과체중 : 23.0 ~ 24.9
 - 비만 : >= 25.0
'''

def bmi_category(bmi):
    if bmi < 18.5:
        return "저체중"
    elif bmi < 23.0:
        return "정상"
    elif bmi < 25.0:
        return "과체중"
    else:
        return "비만"

df['BMI분류'] = df['BMI'].apply(bmi_category)
print("[BMI 분류 결과]")
print(df['BMI분류'].head())

#====================================
# [4] 분석
#====================================
print("===================================")
print("[4] 분석")
print("====================================")

# 4-1) 최고/최저 : BMI가 가장 높은/낮은 사람
max_bmi_row = df.loc[df['BMI'].idxmax()] # 최대값 위치 변환
min_bmi_row = df.loc[df['BMI'].idxmin()] # 최솟값 위치 변환

print("\n[BMI 최고]")
print(max_bmi_row[['환자ID', '성별', '나이', '키_cm', '몸무게_kg', 'BMI', 'BMI분류']])
print("\n[BMI 최저]")
print(min_bmi_row[['환자ID', '성별', '나이', '키_cm', '몸무게_kg', 'BMI', 'BMI분류']])

# 4-2) 혈압 분류별 평균 지표
# groupby() : 데이터를 같은 기준으로 묶는 기능
bp_group = df.groupby('혈압분류')[['SBP(수축기)', 'DBP(이완기)', '맥박압_PP', '평균동맥압_MAP', 'BMI', '운동빈도(주)']].mean()

print("\n[혈압 분류별 평균]")
print(bp_group)

# 4-3) 성별 평균 지표
gender_mean = df.groupby('성별')[['SBP(수축기)', 'DBP(이완기)', '맥박압_PP', '평균동맥압_MAP', 'BMI', '운동빈도(주)']].mean()

print("\n[성별 평균 비교]")
print(gender_mean)

# 4-4) 나이대(연령대) 만들기 + 그룹 평균
'''
나이대는 분석에서 자주 사용
 - pd.cut() : 숫자를 구간으로 나눠 범주형 컬럼 생성
'''
bins = [0, 29, 39, 49, 59, 120]
labels = ['20대', '30대', '40대', '50대', '60대이상']
df['나이대'] = pd.cut(df['나이'], bins=bins, labels=labels, right=True)

age_group = df.groupby('나이대')[['SBP(수축기)', 'DBP(이완기)', '공복혈당', 'BMI']].mean()
print("\n[나이대별 평균]")
print(age_group)

# 4-5) 상관관계(수치형 컬럼)
'''
corr() : 상관계수(-1 ~ 1)
 - 1에 가까울수록 같이 오르는 경향(양의 상관)
 - -1에 가까울수록 반대로 움직이는 경향(음의 상관)
 - 0에 가까우면 관계가 거의 없다(무상관에 가까움)
'''
corr_cols = ['나이', '키_cm', '몸무게_kg', 'SBP(수축기)', 'DBP(이완기)', '공복혈당', '운동빈도(주)', 'BMI', '맥박압_PP', '평균동맥압_MAP']
corr_df = df[corr_cols].corr()
print("\n[상관관계]")
print(corr_df)
print("\n[상관관계 - 소수점 정리]")
print(corr_df.round(2))

# 4-6) 조건 필터링
'''
예시 위험군 조건
 - 위험: 고혈압2기
 - 공복혈당: 126 이상(당뇨 기준으로 자주 언급되는 구간)
 - BMI : 25 이상 (비만)

 해당 조건 중 하나라도 True면 관심 대상이라고 가정
'''
rist_df = df[
    (df['혈압분류'] == "고혈압2기") |
    (df['공복혈당'] >= 126) |
    (df['BMI'] >= 25.0)
].copy()

print("\n[관심 대상 목록]")
print(rist_df[['환자ID', '성별', '나이', 'SBP(수축기)', 'DBP(이완기)', '혈압분류', '공복혈당', 'BMI', 'BMI분류', '운동빈도(주)']])

# 4-7) 교차 분석 : 성별 + 혈압분류
'''
두 개의 기준을 동시에 비교하는 분석
 - 성별에 따라 혈압 분포가 어떻게 다른지 확인
 - 단순 평균보다 더 '분포'를 볼 수 있음
'''

# crosstab() : 두 개 이상의 범주형 변수의 관계를 표로 정리하는 함수
cross = pd.crosstab(df['성별'], df['혈압분류'])

print("\n[성별 + 혈압분류 교차 표]")
print(cross)

# 4-8) 비율 분석
'''
개수보다 비율이 중요할 때가 맣음
 - 전체 주에서 각 혈압분류가 몇 %(백분율)인지 확인
'''
# value_counts() : 각 항목이 몇 번 나왔는지 세는 함수
# normalize=True를 넣으면 개수가 아니라 비율로 바꿔줌
# ex) 정상: 3/5 = 0.6
# * 100은 비율이 보통 0~1사이 소수로 나오기 때문에 쉽게 %로 바꾸기 위해
bp_ratio = df['혈압분류'].value_counts(normalize=True)*100

print("\n[혈압분류 비율 (%)]")
print(bp_ratio.round(1))

# 4-9) 상위 N명 분석(Top 분석)
'''
특정 기준에서 상위 데이터 확인
 - BMI 높은 사람 상위 3명
 - 위험군 빠르게 파악 가능
'''
# nlargest() : 데이터에서 값이 가장 큰 상위 N개를 뽑는 함수
top3_bmi = df.nlargest(3, 'BMI')

print("\n[BMI 상위 3명]")
print(top3_bmi[['환자ID', 'BMI', 'BMI분류']])

# 4-10) 조건 + 그룹 분석
'''
특정 조건을 만족하는 데이터만 따로 분석
 - 고혈압2기 환자만 따로 모아서 성별 평균 비교
'''

ht_df = df[df['혈압분류'] == '고혈압2기']
ht_mean = ht_df.groupby('성별')[['BMI', '공복혈당']].mean()

print("\n[고혈압2기 환자 - 성별 평균]")
print(ht_mean)

# 4-11) 순위(Rank) 만들기
'''
데이터에 순서를 부여
 - BMI 기준 순위 작성
'''
# rank() : 데이터에 순위를 매기는 함수
# ascending=True는 작은 값부터 1등으로 순위를 부여함
df['BMI순위'] = df['BMI'].rank(ascending=True)

print("\n[BMI 순위](환자ID 기준 순서)")
print(df[['환자ID', 'BMI', 'BMI순위']])

# BMI 순위별로 출력
print("\n[BMI 순위](BMI순위 기준 순서)")
df_sorted = df.sort_values('BMI순위')
print(df_sorted[['환자ID', 'BMI', 'BMI순위']])

# 4-12) 위험 점수 만들기
'''
조건을 점수화해서 위험도를 표현

조건:
 - 고혈압2기
 - 공복혈당 >= 126
 - BMI >= 25
 - 각 조건마다 1점
'''

df['위험점수'] = (
    (df['혈압분류'] == '고혈압2기').astype(int) +
    (df['공복혈당'] >= 126).astype(int) +
    (df['BMI'] >= 25).astype(int)
)

print("\n[위험점수]")
print(df[['환자ID', '위험점수']])

#====================================
# [5] 시각화
#====================================
print("===================================")
print("[5] 시각화")
print("====================================")

# 5-1) 혈압(SBP/DBP) 분포 히스토그램
'''
히스토그램
 - 숫자 데이터가 어느 구간에 얼마나 많이 모여 있는지 보여주는 그래프
 - bin이 크면 클 수록 나뉘는 구간이 촘촘함
 - 구간 수에 따라 그래프 모양이 달라짐
'''
# # Linux 한글 폰트 설정
# plt.rcParams['font.family'] = 'NanumGothic'

# # 음수 기호 깨짐 방지
# plt.rcParams['axes.unicode_minus'] = False

plt.figure() # 그래프 공간 생성
plt.hist(df['SBP(수축기)'], bins=8) # 구간 설정
# plt.title('수축기 혈압(SBP) 분포') # 그래프 위 제목
plt.title('SBP DIST') # 그래프 위 제목 (수축기 혈압(SBP) 분포)
plt.xlabel('SBP(mmHg)') # 가로축 의미 설명
# plt.ylabel('빈도') # 세로축 의미 설명
plt.ylabel('frequency') # 세로축 의미 설명(빈도)
plt.tight_layout() # 글자가 잘리지 않게 자동 정렬
plt.show()

plt.figure() # 그래프 공간 생성
plt.hist(df['DBP(이완기)'], bins=8) # 구간 설정
plt.title('DBP DIST') # 그래프 위 제목 (이완기 혈압(DBP) 분포)
plt.xlabel('DBP(mmHg)') # 가로축 의미 설명
plt.ylabel('frequency') # 세로축 의미 설명(빈도)
plt.tight_layout() # 글자가 잘리지 않게 자동 정렬
plt.show()

# 5-2) 혈압분류별 평균 SBP 막대그래프
