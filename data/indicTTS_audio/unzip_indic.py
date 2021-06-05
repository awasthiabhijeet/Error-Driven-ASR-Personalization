import argparse
import os
import zipfile

def unzip_file(data_folder, file):
	with zipfile.ZipFile(os.path.join(data_folder,file),'r') as zip_ref:
		zip_ref.extractall(os.path.join(
			data_folder,file.strip().split()[-1][:-4]
		))

unzip_mode = 'file'

parser=argparse.ArgumentParser(description='unzipping indicTTS data')
parser.add_argument('--input_path',required=True)
args=parser.parse_args()

if unzip_mode == 'folder':
	data_folder=args.input_path

	for file in os.listdir(data_folder):
		if 'english.zip' in file:
			unzip_file(data_folder,file)

else:
	zipfile_path = args.input_path
	data_folder, file = os.path.split(zipfile_path)
	unzip_file(data_folder, file)


		

