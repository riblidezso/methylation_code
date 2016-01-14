import time
import subprocess

def run_sqlilte3(command,db,output=''):
	"""
	Execute sql query on database.

	Output written to filename if given.
	Note: Using the sqlite3 command line tool with a temp file.
		This way the query can be interrupted without killing the python kernel.
	Note: /usr/bin/sqlite3 is 3-4 times faster than anaconda sqlite3.

	"""
	start=time.time()
	with open('tempf.sql','w') as tempf:
		tempf.write(command)
		
	if output != '':
		output=' > '+output
	
	try:
		print subprocess.check_output('/usr/bin/sqlite3 '+ db + ' < tempf.sql '+ output,
									  shell=True, stderr=subprocess.STDOUT)
	except subprocess.CalledProcessError, e:
		print e.output
	
	subprocess.call(['rm','tempf.sql'])
	print 'It took',int(time.time()-start),'s'
