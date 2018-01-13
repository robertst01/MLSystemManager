// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

using System;
using System.IO;
using MLSystemManager.Algorithms;

namespace MLSystemManager
{
    public class SystemManager
	{
		public SupervisedLearner GetLearner(Parameters parameters, Random rand)
		{
			switch (parameters.Learner)
			{
				case "baseline":
					return new BaselineLearner();
				case "perceptron":
					return new Perceptron(parameters);
				case "neuralnet":
					return new NeuralNet(parameters);
				case "decisiontree":
					return new DecisionTree(parameters);
				case "knn":
					return new InstanceBasedLearner(parameters);
				case "clustering":
					return new Clustering(parameters);
				case "dropout":
					return new Dropout(parameters);
				case "bptt":
					if ((parameters.Hidden != null) && (parameters.Hidden.Length > 1))
					{
						throw new Exception("Only one hidden layer is supported");
					}
					return new BPTT(parameters);
				case "dbn":
					return new DBN(parameters);
				case "sae":
					return new SAE(parameters);
				case "relu":
					return new ReLU(parameters);
				case "som":
					return new SOM(parameters);
				case "smlp":
					return new SMLP(parameters);
				case "amlp":
					return new AMLP(parameters);
				case "dmlp":
					return new DMLP(parameters);
				case "rnn":
					return new RNN(parameters);
				case "nnrev":
					return new NNRev(parameters);
				default:
					throw new Exception("Unrecognized model: " + parameters.Learner);
			}
		}

		public void Run(string[] args)
		{
			//args = new string[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

			var rand = Rand.Get();

			//Parse the command line arguments
			ArgParser.Parse(args);
			var parameters = Parameters.Get(ArgParser.ParameterFile);

			var fileName = parameters.Arff; //File specified by the user
			var learnerName = parameters.Learner; //Learning algorithm specified by the user
			var evalMethod = parameters.Evaluation; //Evaluation method specified by the user
			var evalParameter = parameters.EvalExtra; //Evaluation parameters specified by the user
			var rate = parameters.Rate;
			var outputs = parameters.Outputs;
			var outputFileName = parameters.OutputFileName;
			var weightsFileName = parameters.WeightsFileName;
			var verbose = parameters.Verbose;
			var normalize = parameters.Normalize;

			// Load the model
			var learner = GetLearner(parameters, rand);

			learner.Verbose = verbose;

			if (!string.IsNullOrEmpty(outputFileName))
			{
				learner.OutputFileName = outputFileName;
			}

			if (!string.IsNullOrEmpty(weightsFileName))
			{
				learner.WeightsFileName = weightsFileName;
			}

			// Load the ARFF file
			var data = new Matrix();
			data.LoadArff(fileName);

			if (outputs > data.Cols() - 1)
			{
				Console.WriteLine("Too many outputs: " + outputs);
				Environment.Exit(0);
			}

			if (normalize)
			{
				Console.WriteLine("Using normalized data\n");
				data.Normalize();
			}

			// Print some stats
			Console.WriteLine();
			Console.WriteLine("Dataset name: " + fileName);
			Console.WriteLine("Number of instances: " + data.Rows());
			Console.WriteLine("Number of attributes: " + data.Cols());
			Console.WriteLine("Learning algorithm: " + learnerName);
			Console.WriteLine("Evaluation method: " + evalMethod);
			Console.WriteLine("Learning Rate: " + rate);
			Console.WriteLine("Outputs: " + outputs);
			Console.WriteLine();

			if (!string.IsNullOrEmpty(learner.OutputFileName))
			{
				using (var w = new StreamWriter(learner.OutputFileName))
				{
					w.WriteLine("Dataset name: " + fileName);
					w.WriteLine("Number of instances: " + data.Rows());
					w.WriteLine("Number of attributes: " + data.Cols());
					w.WriteLine("Learning algorithm: " + learnerName);
					w.WriteLine("Evaluation method: " + evalMethod);
					w.WriteLine("Learning Rate: " + rate);
					w.WriteLine("Outputs: " + outputs);
					if (normalize)
					{
						w.WriteLine("Using normalized data");
					}
					w.WriteLine();
				}
			}

			if (evalMethod.Equals("training"))
			{
				Console.WriteLine("Calculating accuracy on training set...");
				var features = new VMatrix(data, 0, 0, data.Rows(), data.Cols() - outputs);
				var labels = new VMatrix();
				if (outputs > 0)
				{
					labels = new VMatrix(data, 0, data.Cols() - outputs, data.Rows(), outputs);
				}
				var confusion = new Matrix();
				var startTime = DateTime.Now.Ticks;
				var colMin = new double[data.Cols()];
				var colMax = new double[data.Cols()];
				for (var col = 0; col < data.Cols(); col++)
				{
					colMin[col] = data.ColumnMinOrig(col);
					colMax[col] = data.ColumnMaxOrig(col);
				}
				learner.VTrain(features, labels, colMin, colMax);
				var elapsedTime = new TimeSpan(DateTime.Now.Ticks - startTime);
				Console.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
				if (outputs > 0)
				{
					var accuracy = learner.VMeasureAccuracy(features, labels, confusion);
					Console.WriteLine("Training set accuracy: " + accuracy);

					if (!string.IsNullOrEmpty(learner.OutputFileName))
					{
						using (var w = File.AppendText(learner.OutputFileName))
						{
							w.WriteLine();
							w.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
							w.WriteLine("Training set accuracy: " + accuracy);
						}
					}
				}
                
				if (verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (evalMethod.Equals("static"))
			{
				var testData = new Matrix();
				testData.LoadArff(evalParameter);
				if (normalize)
				{
					testData.Normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
				}

				Console.WriteLine("Calculating accuracy on separate test set...");
				Console.WriteLine("Test set name: " + evalParameter);
				Console.WriteLine("Number of test instances: " + testData.Rows());
				var features = new VMatrix(data, 0, 0, data.Rows(), data.Cols() - outputs);
				var labels = new VMatrix(data, 0, data.Cols() - outputs, data.Rows(), outputs);
				var startTime = DateTime.Now.Ticks;
				var colMin = new double[data.Cols()];
				var colMax = new double[data.Cols()];
				for (var col = 0; col < data.Cols(); col++)
				{
					colMin[col] = data.ColumnMinOrig(col);
					colMax[col] = data.ColumnMaxOrig(col);
				}
				learner.VTrain(features, labels, colMin, colMax);
				var elapsedTime = new TimeSpan(DateTime.Now.Ticks - startTime);
				Console.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
				var trainAccuracy = learner.VMeasureAccuracy(features, labels, null);
				Console.WriteLine("Training set accuracy: " + trainAccuracy);
				var testFeatures = new VMatrix(testData, 0, 0, testData.Rows(), testData.Cols() - outputs);
				var testLabels = new VMatrix(testData, 0, testData.Cols() - outputs, testData.Rows(), outputs);
				var confusion = new Matrix();
				var testAccuracy = learner.VMeasureAccuracy(testFeatures, testLabels, confusion);
				Console.WriteLine("Test set accuracy: " + testAccuracy);
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (var w = File.AppendText(learner.OutputFileName))
					{
						w.WriteLine();
						w.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
						w.WriteLine("Training set accuracy: " + trainAccuracy);
						w.WriteLine("Test set accuracy: " + testAccuracy);
					}
				}
				if (verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (evalMethod.Equals("random"))
			{
				Console.WriteLine("Calculating accuracy on a random hold-out set...");
				var trainPercent = Double.Parse(evalParameter);
				if (trainPercent < 0 || trainPercent > 1)
				{
					throw new Exception("Percentage for random evaluation must be between 0 and 1");
				}
				Console.WriteLine("Percentage used for training: " + trainPercent);
				Console.WriteLine("Percentage used for testing: " + (1 - trainPercent));
				var vData = new VMatrix(data, 0, 0, data.Rows(), data.Cols());
				if (!(learner is BPTT))
				{
					vData.Shuffle(rand);
				}
				var trainSize = (int)(trainPercent * vData.Rows());
				var trainFeatures = new VMatrix(vData, 0, 0, trainSize, vData.Cols() - outputs);
				var trainLabels = new VMatrix(vData, 0, vData.Cols() - outputs, trainSize, outputs);
				var testFeatures = new VMatrix(vData, trainSize, 0, vData.Rows() - trainSize, vData.Cols() - outputs);
				var testLabels = new VMatrix(vData, trainSize, vData.Cols() - outputs, vData.Rows() - trainSize, outputs);
				var startTime = DateTime.Now.Ticks;
				var colMin = new double[data.Cols()];
				var colMax = new double[data.Cols()];
				for (var col = 0; col < data.Cols(); col++)
				{
					colMin[col] = vData.ColumnMinOrig(col);
					colMax[col] = data.ColumnMaxOrig(col);
				}
				learner.VTrain(trainFeatures, trainLabels, colMin, colMax);
				var elapsedTime = new TimeSpan(DateTime.Now.Ticks - startTime);
				Console.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
				var trainAccuracy = learner.VMeasureAccuracy(trainFeatures, trainLabels, null);
				Console.WriteLine("Training set accuracy: " + trainAccuracy);
				var confusion = new Matrix();
				var testAccuracy = learner.VMeasureAccuracy(testFeatures, testLabels, confusion);
				Console.WriteLine("Test set accuracy: " + testAccuracy);
				var testMSE = learner.VGetMSE(testFeatures, testLabels);
				Console.WriteLine("Test set MSE: " + testMSE);
                
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (var w = File.AppendText(learner.OutputFileName))
					{
						w.WriteLine();
						w.WriteLine("Percentage used for training: " + trainPercent);
						w.WriteLine("Percentage used for testing: " + (1 - trainPercent));
						w.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
						w.WriteLine("Training set accuracy: " + trainAccuracy);
						w.WriteLine("Test set accuracy: " + testAccuracy);
						w.WriteLine("Test set MSE: " + testMSE);
					}
				}
                
				if (verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (evalMethod.Equals("cross"))
			{
				Console.WriteLine("Calculating accuracy using cross-validation...");
				var folds = int.Parse(evalParameter);
				if (folds <= 0)
				{
					throw new Exception("Number of folds must be greater than 0");
				}
				Console.WriteLine("Number of folds: " + folds);
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (var w = File.AppendText(learner.OutputFileName))
					{
						w.WriteLine();
						w.WriteLine("Calculating accuracy using cross-validation...");
						w.WriteLine("Number of folds: " + folds);
					}
				}
				var reps = 1;
				var sumAccuracy = 0.0;
				long ticks = 0;
				for (var j = 0; j < reps; j++)
				{
					data.Shuffle(rand);
					for (var i = 0; i < folds; i++)
					{
						var begin = i * data.Rows() / folds;
						var end = (i + 1) * data.Rows() / folds;
						var trainFeatures = new Matrix(data, 0, 0, begin, data.Cols() - outputs);
						var trainLabels = new Matrix(data, 0, data.Cols() - outputs, begin, outputs);
						var testFeatures = new Matrix(data, begin, 0, end - begin, data.Cols() - outputs);
						var testLabels = new Matrix(data, begin, data.Cols() - outputs, end - begin, outputs);
						trainFeatures.Add(data, end, 0, data.Rows() - end);
						trainLabels.Add(data, end, data.Cols() - outputs, data.Rows() - end);
						var startTime = DateTime.Now.Ticks;
						var colMin = new double[data.Cols()];
						var colMax = new double[data.Cols()];
						for (var col = 0; col < data.Cols(); col++)
						{
							colMin[col] = data.ColumnMinOrig(col);
							colMax[col] = data.ColumnMaxOrig(col);
						}
						learner.Train(trainFeatures, trainLabels, colMin, colMax);
						ticks = DateTime.Now.Ticks - startTime;
						var accuracy = learner.MeasureAccuracy(testFeatures, testLabels, null);
						sumAccuracy += accuracy;
						Console.WriteLine("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
						if (!string.IsNullOrEmpty(learner.OutputFileName))
						{
							using (var w = File.AppendText(learner.OutputFileName))
							{
								w.WriteLine("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
								w.WriteLine();
							}
						}
					}
				}
				ticks /= (reps * folds);
				var elapsedTime = new TimeSpan(ticks);
				Console.WriteLine("Average time to train (in seconds): " + elapsedTime.TotalSeconds);
				Console.WriteLine("Mean accuracy=" + (sumAccuracy / (reps * folds)));
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (var w = File.AppendText(learner.OutputFileName))
					{
						w.WriteLine();
						w.WriteLine("Average time to train (in seconds): " + elapsedTime.TotalSeconds);
						w.WriteLine("Mean accuracy=" + (sumAccuracy / (reps * folds)));
					}
				}
			}
		}
	}
}