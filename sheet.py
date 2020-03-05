import xlwt
import xlrd
from xlutils.copy import copy
import os
import datetime
st_name = 'Aashish'
def mark_present(st_name):

	names = os.listdir('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/output/')
	print(names)

	sub = 'HACKATHON'

	rb = xlrd.open_workbook('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/attendance/' + sub + '.xlsx')
	wb = copy(rb)
	sheet = wb.get_sheet(0)
	sheet.write(1,1,str(datetime.datetime.now()))


	count = 2
	for i in names:
	    if i in st_name:
              sheet.write(count, 1, 'P')
	    else:
              sheet.write(count, 1, 'A')
	    sheet.write(count, 0, i)
	    count += 1

	wb.save('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/attendance/' + sub + '.xlsx')


mark_present(st_name)
