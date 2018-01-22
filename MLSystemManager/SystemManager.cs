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
			var parameters = Parameters.Get();

			// Load the model
			var learner = GetLearner(parameters, rand);

			// Load the ARFF file
			var data = new Matrix();
			data.LoadArff(parameters.Arff);

			if (parameters.Outputs > data.Cols() - 1)
			{
				Console.WriteLine("Too many outputs: " + parameters.Outputs);
				Environment.Exit(0);
			}

			if (parameters.Normalize)
			{
				Console.WriteLine("Using normalized data\n");
				data.Normalize();
			}

			// Print some stats
			Console.WriteLine();
			Console.WriteLine("Dataset name: " + parameters.Arff);
			Console.WriteLine("Number of instances: " + data.Rows());
			Console.WriteLine("Number of attributes: " + data.Cols());
			Console.WriteLine("Learning algorithm: " + parameters.Learner);
			Console.WriteLine("Evaluation method: " + parameters.Evaluation);
			Console.WriteLine("Learning Rate: " + parameters.Rate);
			Console.WriteLine("Outputs: " + parameters.Outputs);
			Console.WriteLine("Snapshot File: " + parameters.SnapshotFileName);
			Console.WriteLine();

			if (parameters.Evaluation == "training")
			{
				Console.WriteLine("Calculating accuracy on training set...");
				var features = new VMatrix(data, 0, 0, data.Rows(), data.Cols() - parameters.Outputs);
				var labels = new VMatrix();
				if (parameters.Outputs > 0)
				{
					labels = new VMatrix(data, 0, data.Cols() - parameters.Outputs, data.Rows(), parameters.Outputs);
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
				if (parameters.Outputs > 0)
				{
					var accuracy = learner.VMeasureAccuracy(features, labels, confusion);
					Console.WriteLine("Training set accuracy: " + accuracy);
				}
                
				if (parameters.Verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (parameters.Evaluation == "static")
			{
				var testData = new Matrix();
				testData.LoadArff(parameters.EvalExtra);
				if (parameters.Normalize)
				{
					testData.Normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
				}

				Console.WriteLine("Calculating accuracy on separate test set...");
				Console.WriteLine("Test set name: " + parameters.EvalExtra);
				Console.WriteLine("Number of test instances: " + testData.Rows());
				var features = new VMatrix(data, 0, 0, data.Rows(), data.Cols() - parameters.Outputs);
				var labels = new VMatrix(data, 0, data.Cols() - parameters.Outputs, data.Rows(), parameters.Outputs);
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
				var testFeatures = new VMatrix(testData, 0, 0, testData.Rows(), testData.Cols() - parameters.Outputs);
				var testLabels = new VMatrix(testData, 0, testData.Cols() - parameters.Outputs, testData.Rows(), parameters.Outputs);
				var confusion = new Matrix();
				var testAccuracy = learner.VMeasureAccuracy(testFeatures, testLabels, confusion);
				Console.WriteLine("Test set accuracy: " + testAccuracy);
				if (parameters.Verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (parameters.Evaluation == "test")
			{
				var testData = new Matrix();
				testData.LoadArff(parameters.EvalExtra);
				if (parameters.Normalize)
				{
					testData.Normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
				}

				Console.WriteLine("Calculating accuracy on separate test set...");
				Console.WriteLine("Test set name: " + parameters.EvalExtra);
				Console.WriteLine("Number of test instances: " + testData.Rows());
				var testFeatures = new VMatrix(testData, 0, 0, testData.Rows(), testData.Cols() - parameters.Outputs);
				var testLabels = new VMatrix(testData, 0, testData.Cols() - parameters.Outputs, testData.Rows(), parameters.Outputs);
				var confusion = new Matrix();
				var testAccuracy = learner.VMeasureAccuracy(testFeatures, testLabels, confusion);
				Console.WriteLine("Test set accuracy: " + testAccuracy);
				if (parameters.Verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (parameters.Evaluation == "random")
			{
				Console.WriteLine("Calculating accuracy on a random hold-out set...");
				var trainPercent = double.Parse(parameters.EvalExtra);
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
				var trainFeatures = new VMatrix(vData, 0, 0, trainSize, vData.Cols() - parameters.Outputs);
				var trainLabels = new VMatrix(vData, 0, vData.Cols() - parameters.Outputs, trainSize, parameters.Outputs);
				var testFeatures = new VMatrix(vData, trainSize, 0, vData.Rows() - trainSize, vData.Cols() - parameters.Outputs);
				var testLabels = new VMatrix(vData, trainSize, vData.Cols() - parameters.Outputs, vData.Rows() - trainSize, parameters.Outputs);
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
                
				if (parameters.Verbose)
				{
					Console.WriteLine("\nConfusion matrix: (Row=target value, Col=predicted value)");
					confusion.Print();
					Console.WriteLine("\n");
				}
			}
			else if (parameters.Evaluation == "cross")
			{
				Console.WriteLine("Calculating accuracy using cross-validation...");
				var folds = int.Parse(parameters.EvalExtra);
				if (folds <= 0)
				{
					throw new Exception("Number of folds must be greater than 0");
				}
				Console.WriteLine("Number of folds: " + folds);
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
						var trainFeatures = new Matrix(data, 0, 0, begin, data.Cols() - parameters.Outputs);
						var trainLabels = new Matrix(data, 0, data.Cols() - parameters.Outputs, begin, parameters.Outputs);
						var testFeatures = new Matrix(data, begin, 0, end - begin, data.Cols() - parameters.Outputs);
						var testLabels = new Matrix(data, begin, data.Cols() - parameters.Outputs, end - begin, parameters.Outputs);
						trainFeatures.Add(data, end, 0, data.Rows() - end);
						trainLabels.Add(data, end, data.Cols() - parameters.Outputs, data.Rows() - end);
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
					}
				}
				ticks /= (reps * folds);
				var elapsedTime = new TimeSpan(ticks);
				Console.WriteLine("Average time to train (in seconds): " + elapsedTime.TotalSeconds);
				Console.WriteLine("Mean accuracy=" + (sumAccuracy / (reps * folds)));
			}
		}
	}
}