# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:09:19 2020

@author: Shibu Meher
"""

# Extract Information from Vincent University Dataset
# Making a function to extract data
from pyedflib import highlevel
import re
import numpy as np

def extract_data(filename1, filename2):
  signals, signal_headers, header = highlevel.read_edf(filename1)
  EEG_ECG = np.asarray([signals[3], signals[4], signals[5]])
  # Function to label apnea state
  def label_apnea(s):
    if s=="HYP-C":
      return 1
    elif s=="HYP-O":
      return 2
    elif s=="HYP-M":
      return 3
    elif s=="APNEA-C":
      return 4
    elif s=="APNEA-O":
      return 5
    elif s=="APNEA-M":
      return 6
    elif s=="Sleep":
      return 7
    else:
      print("Give proper input")
      return 0
  
  # Function to read the respiratory event
  def read_events(lines):
    event_data = []
    for i in range(3,len(lines)-1):
      a = re.split("\s+", lines[i])
      time = list(map(int,a[0].split(":")))
      typ = label_apnea(a[1])
      dur = int(a[2])
      data = [time, typ, dur]
      event_data.append(data)
    return event_data
  
  f = open(filename2, 'r')
  lines = f.readlines()
  event_data = read_events(lines)
  f.close()

  # Start time extractor
  def start_time_calc(header):
    start_time = [header['startdate'].hour, header['startdate'].minute, header['startdate'].second]
    if start_time[0] > 12:
      start_time[0]=start_time[0]-24
    return start_time

  b = start_time_calc(header)
  print(b)
  final_ind = -1
  event_info = []
  for i in range(len(event_data)):
    a = event_data[i][0]
    c = [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    initial_ind = c[0]*3600*128+c[1]*60*128+c[2]*128
    info = [[final_ind+1, initial_ind-1], 7]
    event_info.append(info)
    final_ind = initial_ind + event_data[i][2]*128
    info = [[initial_ind, final_ind], event_data[i][1]]
    event_info.append(info)

  info = [[final_ind+1, len(signals[3])-1], 7]
  event_info.append(info)

  def embed_data(EEG_ECG, event_info):
    C3A2 = []
    C4A1 = []
    ECG = []
    for i in range(len(event_info)):
      C3A2.append([EEG_ECG[0][event_info[i][0][0]:event_info[i][0][1]],event_info[i][1]])
      C4A1.append([EEG_ECG[1][event_info[i][0][0]:event_info[i][0][1]],event_info[i][1]])
      ECG.append([EEG_ECG[2][event_info[i][0][0]:event_info[i][0][1]],event_info[i][1]])
  
    return [C3A2, C4A1, ECG]

  embedded_data = embed_data(EEG_ECG, event_info)
  
  return [EEG_ECG, event_info, embedded_data]


filename1 = "ucddb008.rec"
filename2 = "ucddb008_respevt.txt"

[EEG_ECG, event_info, embedded_data] = extract_data(filename1, filename2)