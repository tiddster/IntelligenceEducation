import pandas as pd

root_path = "D:\新建文件夹\Dataset\DKT\\train.csv"
test_path = "D:\新建文件夹\Dataset\DKT\\test.csv"

train_data = pd.read_csv(root_path)
test_data = pd.read_csv(test_path)

studentList = list(train_data["student"])
skillList = list(train_data["skill"])
questionList = list(train_data["question"])
correctList = list(train_data["correctness"])

"""
首先判断学生的学习能力: 大致符合正态分布
"""
student2ability = {s:{'r':0, 't':0, 'c':0} for s in set(studentList)}
for s,c in zip(studentList, correctList):
    student2ability[s]['t'] += 1
    if c == 1:
        student2ability[s]['r'] += 1
    student2ability[s]['c'] = student2ability[s]['r']/student2ability[s]['t']

extrs_stu, good_stu, soso_stu, bad_stu, poor_stu = 0, 0, 0, 0, 0
for i in range(len(student2ability)):
    s = student2ability[i]['c']
    if s >= 0.9:
        extrs_stu += 1
    elif s >= 0.7:
        good_stu += 1
    elif s >= 0.5:
        soso_stu += 1
    elif s >= 0.3:
        bad_stu += 1
    else:
        poor_stu += 1

print(extrs_stu, good_stu, soso_stu, bad_stu, poor_stu)

"""
判断知识的难度：大致符合正太分布
"""
skill2diff = {sk: {'r':0, 't':0, 'c':0} for sk in set(skillList)}
for sk, c in zip(skillList, correctList):
    skill2diff[sk]['t'] += 1
    if c == 1:
        skill2diff[sk]['r'] += 1
    skill2diff[sk]['c'] = skill2diff[sk]['r']/skill2diff[sk]['t']

easy_skill, soso_skill, diff_skill = 0, 0, 0
for i in range(len(skill2diff)):
    s = skill2diff[i]['c']
    if s >= 0.7:
        easy_skill += 1
    elif s >= 0.4:
        soso_skill += 1
    else:
        diff_skill += 1
print(easy_skill, soso_skill, diff_skill)

"""
判断问题的难度：大致符合正太分布
"""
question2diff = {q:{'r':0, 't':0, 'c':0} for q in set(questionList)}
for q, c in zip(questionList, correctList):
    question2diff[q]['t'] += 1
    if c == 1:
        question2diff[q]['r'] += 1
    question2diff[q]['c'] = question2diff[q]['r']/question2diff[q]['t']

easy_ques, soso_ques, diff_ques = 0, 0, 0
for i in range(len(skill2diff)):
    s = question2diff[i]['c']
    if s >= 0.7:
        easy_ques += 1
    elif s >= 0.3:
        soso_ques += 1
    else:
        diff_ques += 1
print(easy_ques, soso_ques, diff_ques)

