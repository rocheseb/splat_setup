"""
Create a splat control file based on an input template modified with command line arguments
"""

import os
import sys
import numpy as np

# template for modifying input file
from jinja2 import Template
import json
import argparse


def csl_u(x):
	"""
	comma separated list (uppercase)
	"""
	return [i.upper() for i in x.split(',')]


def update_dict_from_json(x,json_file):
	with open(json_file,'rb') as f:
		json_inputs = json.load(f)
	for key,val in json_inputs.items():
		if (key not in x) or (x[key] is None): # if an argument is given in command line, it is used instead of the json file
			x[key] = val
	return x


def control_setup(**kwargs):
	required_args = ['run_dir','template','json','prf_species','col_species','abs_species','xsec']
	for arg in required_args:
		if arg not in kwargs:
			raise Exception(f'Missing argument: {arg}')
	outdir = os.path.join(kwargs['run_dir'],'run_setup')
	if not os.path.exists(outdir):
		print(f'Creating {outdir}')
		os.mkdir(outdir)

	template_inputs = {}
	if kwargs['json'] is not None:
		update_dict_from_json(kwargs,kwargs['json'])

	PS = '' if 'O2' in kwargs['prf_species']+kwargs['col_species'] else '_PS'
	kwargs['fname'] = f"{kwargs['abs_species'][0]}_{kwargs['xsec']}{PS}_o{kwargs['win_poly_order']}_nfl{kwargs['nfl']}"
	kwargs['outfile'] = os.path.join(outdir,f"{kwargs['fname']}.nc")
	kwargs['log'] = os.path.join(outdir,f"{kwargs['fname']}.log")

	template_inputs.update(kwargs)

	with open(kwargs['template'],'r') as f:
		template = Template(f.read())

	with open('xsec.json','rb') as f:
		xsec_data = json.load(f)
	template_inputs.update(xsec_data[kwargs['xsec']])

	control_file = os.path.join(outdir,f"{kwargs['fname']}.control")
	with open(control_file,'w') as f:
		f.write(template.render(**template_inputs))
	print(f'Creating {control_file}')

	return kwargs


def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('run_dir',help='full path to the splat run directory')
	parser.add_argument('data_dir',help='full path to the data directory for the splat run')
	parser.add_argument('-t','--template',default='/n/home11/sroche/gitrepos/methanesat/template.control')
	parser.add_argument('-j','--json',default=None,help='full path to input json file')
	parser.add_argument('-x','--xsec',default='ori',help='one of the keys of xsec.json to setup the cross section files')
	parser.add_argument('--abs-species',type=csl_u,help='Comma-separated list of gas names to include in spectrum calculation',default=None)
	parser.add_argument('--prf-species',type=csl_u,help='Comma-separated list of gas names to retrieve as profiles',default=None)
	parser.add_argument('--col-species',type=csl_u,help='Comma-separated list of gas names to retrieve as columns',default=None)
	parser.add_argument('--win-poly-order',help='polynomial order for fitting the continuum',default=None)
	parser.add_argument('--l1-rad',help='full path to the L1b radiance file',default=None)
	parser.add_argument('--l2-met',help='full path to the met file for l2 processing',default=None)
	parser.add_argument('--nfl',type=int,help='Number of fine layers for XS calc',default=None)
	args = parser.parse_args()

	path_list = [args.run_dir,args.data_dir,'xsec.json']
	if args.json:
		path_list += [args.json]
	for path in path_list:
		if not os.path.exists(path):
			raise Exception(f'Wong path: {path}')
	
	arguments = control_setup(**args.__dict__)


if __name__=="__main__":
	main()
