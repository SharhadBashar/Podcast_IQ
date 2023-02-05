import sys

from train import Train
from predict import Predict

if __name__ == '__main__':
	try:
		command = sys.argv[1].lower()
	except IndexError:
		print('No command found. Please type train or t for training, or predict or p for predictions')
		exit()
	try:
		path = sys.argv[2]
	except IndexError:
		print('No path to file provided')
		exit()

	if (command == 'train' or sys.argv[1] == 't'):
		print('Starting Training')
		Train(path)
		print('Finished Training')
	elif (command == 'predict' or sys.argv[1] == 'p'):
		print('Starting Predictions')
		Predict(path)
		print('Predictions complete')
	else:
		print('Command {} not understood. Please type train or t for training, or predict or p for predictions'.format(command))
