import xlwt
import xlrd
from xlutils.copy import copy
import os
import datetime
import xlsxwriter

st_name = 'Aashish'
def mark_present(st_name):

	names = os.listdir('output/')
	print(names)

	sub = 'SAMPLE'
	
	if not os.path.exists('attendance/' + sub + '.xlsx'):
		count = 2
		workbook = xlsxwriter.Workbook('attendance/' + sub + '.xlsx')
		print("Creating Spreadsheet with Title: " + sub)
		sheet = workbook.add_worksheet() 
		for i in names:
		    sheet.write(count, 0, i)
		    count += 1
		workbook.close() 

	rb = xlrd.open_workbook('attendance/' + sub + '.xlsx')
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

	wb.save('attendance/' + sub + '.xlsx')


mark_present(st_name)
