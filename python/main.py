import sys

from train import Train
from predict import Predict

if __name__ == '__main__':
	try:
		command = sys.argv[1].lower()
	except IndexError:
		print('No command found. Please type train or t for training, or predict or p for predictions')
		exit()

	if (command == 'train' or sys.argv[1] == 't'):
		Train()
		print('Finished Training')
	elif (command == 'predict' or sys.argv[1] == 'p'):
		predict = Predict()
		name = input('Please enter podcast name:')
		title = input('Please enter podcast title:')
		inp = predict.clean_data(name, title)
		category = predict.predict(inp)
		print('Podcast "{}" and Title "{}" is of "{}" category'.format(name, title, category))
	else:
		print('Command {} not understood. Please type train or t for training, or predict or p for predictions'.format(command))
