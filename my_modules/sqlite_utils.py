def run_sqlilte3(command,db,output=''):
	"""Execute sql query on database."""
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
