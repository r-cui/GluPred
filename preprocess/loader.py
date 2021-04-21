# Script to load data from xml files.
# Chirath Hettiarachchi
# ANU - OHIOH Group.
# May 2020

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime


def round_minute(date_string, round2min):
    # convert the date to round2min intervals.
    # For simplicity rounded backwords. i.e. 0,1,2,3,4 -> 0
    round_min2 = round2min
    date = datetime.datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    new_min = ((date.minute//round_min2) * round_min2)
    date = date.replace(minute=int(new_min))
    date = date.replace(second=0)
    return date


def get_cgm(root, round2min):
    # cgm data, every 5 min
    glucose = []
    glucose_ts = []
    for type_tag in root.findall('glucose_level/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        glucose.append(value)
        glucose_ts.append(ts)
    cgm_frame = [glucose_ts, glucose]
    cgm_frame = np.array(cgm_frame)
    dataframe_cgm = pd.DataFrame(cgm_frame.T, columns=['ts', 'glucose'])
    return dataframe_cgm


def get_fingerstick(root, round2min):
    # cgm data, every 5 min
    fingerstick = []
    fingerstick_ts = []
    for type_tag in root.findall('finger_stick/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        fingerstick.append(value)
        fingerstick_ts.append(ts)
    fingerstick_frame = [fingerstick_ts, fingerstick]
    fingerstick_frame = np.array(fingerstick_frame)
    dataframe_fingerstick = pd.DataFrame(fingerstick_frame.T, columns=['ts', 'fingerstick'])
    return dataframe_fingerstick


def get_gsr(root, round2min):
    # gsr data, every 5 min, Empatica every 1 min!
    gsr = []
    gsr_ts = []
    for type_tag in root.findall('basis_gsr/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        gsr.append(value)
        gsr_ts.append(ts)
    gsr_frame = [gsr_ts, gsr]
    gsr_frame = np.array(gsr_frame)
    dataframe_gsr = pd.DataFrame(gsr_frame.T, columns=['ts', 'gsr'])
    return dataframe_gsr


def get_hr(root, round2min):
    # hr data, every 5 min. Only for Basis Peak!
    hr = []
    hr_ts = []
    for type_tag in root.findall('basis_heart_rate/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        hr.append(value)
        hr_ts.append(ts)
    hr_frame = [hr_ts, hr]
    hr_frame = np.array(hr_frame)
    dataframe_hr = pd.DataFrame(hr_frame.T, columns=['ts', 'hr'])
    return dataframe_hr


def get_st(root, round2min):
    # hr data, every 5 min. Only for Basis Peak!
    st = []
    st_ts = []
    for type_tag in root.findall('basis_skin_temperature/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        st.append(value)
        st_ts.append(ts)
    st_frame = [st_ts, st]
    st_frame = np.array(st_frame)
    dataframe_st = pd.DataFrame(st_frame.T, columns=['ts', 'st'])
    return dataframe_st


def get_basal(root, round2min):
    basal = []
    basal_ts = []
    for type_tag in root.findall('basal/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        basal.append(value)
        basal_ts.append(ts)
    basal_frame = [basal_ts, basal]
    basal_frame = np.array(basal_frame)
    dataframe_basal = pd.DataFrame(basal_frame.T, columns=['ts', 'basal'])
    return dataframe_basal


def get_temp_basal(root, round2min):
    temp_basal = []
    temp_basal_ts = []
    temp_basal_dur = []
    for type_tag in root.findall('temp_basal/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts_begin')
        ts = round_minute(ts, round2min)

        ts_end = type_tag.get('ts_end')
        ts_end = round_minute(ts_end, round2min)

        #temp_basal_dur.append((ts_end-ts).seconds//60)

        temp_basal_dur.append(ts_end)
        temp_basal.append(value)
        temp_basal_ts.append(ts)
    temp_basal_frame = [temp_basal_ts, temp_basal, temp_basal_dur]
    temp_basal_frame = np.array(temp_basal_frame)
    dataframe_temp_basal = pd.DataFrame(temp_basal_frame.T, columns=['ts', 'temp_basal', 'basal_end'])
    return dataframe_temp_basal


def get_bolus(root, round2min):  # check if always ts_begin and ts_end is equal.
    # bwz_carb_input : Amount of CHO entered in to the bolus calculator.
    bolus = []
    bolus_ts = []
    bolus_end = []
    bolus_dur = []

    for type_tag in root.findall('bolus/event'):
        value = type_tag.get('dose')
        ts = type_tag.get('ts_begin')
        ts = round_minute(ts, round2min)

        ts_end = type_tag.get('ts_end')
        ts_end = round_minute(ts_end, round2min)
        # if ts_end != ts:
        #     print("Time mismatch in Bolus Infusion start and end times.")

        bolus_dur.append((ts_end - ts).seconds // 60)
        bolus_end.append(ts_end)

        bolus.append(value)
        bolus_ts.append(ts)
    bolus_frame = [bolus_ts, bolus, bolus_dur, bolus_end]
    bolus_frame = np.array(bolus_frame)
    dataframe_bolus =pd.DataFrame(bolus_frame.T, columns=['ts', 'bolus', 'bolus_dur', 'bolus_end'])
    return dataframe_bolus


def get_meal(root, round2min):
    # cgm data, every 5 min
    meal = []
    meal_ts = []
    meal_type = []
    for type_tag in root.findall('meal/event'):
        carbs = type_tag.get('carbs')
        ts = type_tag.get('ts')
        type = type_tag.get('type')
        ts = round_minute(ts, round2min)
        meal.append(carbs)
        meal_ts.append(ts)
        meal_type.append(type)
    meal_frame = [meal_ts, meal, meal_type]
    meal_frame = np.array(meal_frame)
    dataframe_meal = pd.DataFrame(meal_frame.T, columns=['ts', 'carbs','meal_type'])
    return dataframe_meal


def get_exercise(root, round2min):
    # cgm data, every 5 min
    exercise = []
    exercise_ts = []
    exercise_dur = []
    for type_tag in root.findall('exercise/event'):
        value = type_tag.get('intensity')
        ts = type_tag.get('ts')
        duration = type_tag.get('duration')
        ts = round_minute(ts, round2min)
        exercise.append(value)
        exercise_ts.append(ts)
        exercise_dur.append(duration)
    exercise_frame = [exercise_ts, exercise, exercise_dur]
    exercise_frame = np.array(exercise_frame)
    dataframe_exercise = pd.DataFrame(exercise_frame.T, columns=['ts', 'exercise_intensity', 'exer_dur'])
    return dataframe_exercise


def get_sleep(root, round2min):
    sleep = []
    sleep_ts = []
    sleep_dur = []
    sleep_end = []
    for type_tag in root.findall('sleep/event'):
        value = type_tag.get('quality')
        ts = type_tag.get('ts_end')  # ts_end and ts_begin swapped issue in dataset.
        ts = round_minute(ts, round2min)

        ts_end = type_tag.get('ts_begin')
        ts_end = round_minute(ts_end, round2min)

        sleep_dur.append((ts_end - ts).seconds // 60)
        sleep.append(value)
        sleep_ts.append(ts)
        sleep_end.append(ts_end)
    bolus_frame = [sleep_ts, sleep, sleep_dur, sleep_end]
    bolus_frame = np.array(bolus_frame)
    dataframe_bolus =pd.DataFrame(bolus_frame.T, columns=['ts', 'sleep', 'sleep_dur', 'sleep_end'])
    return dataframe_bolus


def get_work(root, round2min):
    work = []
    work_ts = []
    work_dur = []
    work_end = []
    for type_tag in root.findall('work/event'):
        value = type_tag.get('intensity')
        ts = type_tag.get('ts_begin')
        ts = round_minute(ts, round2min)

        ts_end = type_tag.get('ts_end')
        ts_end = round_minute(ts_end, round2min)

        work_dur.append((ts_end - ts).seconds // 60)
        work.append(value)
        work_ts.append(ts)
        work_end.append(ts_end)
    bolus_frame = [work_ts, work, work_dur, work_end]
    bolus_frame = np.array(bolus_frame)
    dataframe_bolus =pd.DataFrame(bolus_frame.T, columns=['ts', 'work', 'work_dur', 'work_end'])
    return dataframe_bolus


def load_data(FILE):
    root = ET.parse(FILE).getroot()
    round2min = 5

    # get different data types.
    dataframe_cgm = get_cgm(root, round2min)
    #dataframe_fingerstick = get_fingerstick(root)

    dataframe_gsr = get_gsr(root, round2min)
    dataframe_hr = get_hr(root, round2min)
    dataframe_st = get_st(root, round2min)

    dataframe_basal = get_basal(root, round2min)
    dataframe_bolus = get_bolus(root, round2min)
    dataframe_temp_basal = get_temp_basal(root, round2min)

    dataframe_meal = get_meal(root, round2min)
    dataframe_exercise = get_exercise(root, round2min)
    dataframe_sleep = get_sleep(root, round2min)
    dataframe_work = get_work(root, round2min)

    # set time as the index.
    dataframe_cgm = dataframe_cgm.set_index('ts')
    #dataframe_fingerstick = dataframe_fingerstick.set_index('ts')

    dataframe_gsr = dataframe_gsr.set_index('ts')
    dataframe_hr = dataframe_hr.set_index('ts')
    dataframe_st = dataframe_st.set_index('ts')

    dataframe_basal = dataframe_basal.set_index('ts')
    dataframe_bolus = dataframe_bolus.set_index('ts')
    dataframe_temp_basal = dataframe_temp_basal.set_index('ts')

    dataframe_meal = dataframe_meal.set_index('ts')
    dataframe_exercise = dataframe_exercise.set_index('ts')
    dataframe_sleep = dataframe_sleep.set_index('ts')
    dataframe_work = dataframe_work.set_index('ts')

    data_frames = [dataframe_cgm, dataframe_gsr, dataframe_hr, dataframe_st, dataframe_basal,
                   dataframe_bolus, dataframe_temp_basal, dataframe_meal, dataframe_sleep, dataframe_work, dataframe_exercise ] #, dataframe_fingerstick, ]
    df = data_frames[0]
    for df_ in data_frames[1:]:
        df = df.join(df_,how="outer")
    # print(df.to_csv('temp.csv'))
    return df

