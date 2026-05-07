from infrastructure.train_runner import TrainRunner

if __name__ == "__main__":

	optionPath = './train_options/lite_dvd_downscaled_3.yml'

	trainRunner = TrainRunner(optionPath)

	options = trainRunner.options

	print("\n### Training denoiser model ###")
	print("> Parameters:")
	for key, value in options.items():
		print(f'\t{key}: {value}')
	print('\n')

	trainRunner.train()



