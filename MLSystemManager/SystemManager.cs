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
		/**
			*  When you make a new learning algorithm, you should add a line for it to this method.
			*/
		public SupervisedLearner GetLearner(string model, string learnParameter, Random rand, double rate, double momentum,
			double ph, double pi, int[] hidden, bool prune, bool distance, int k, string ignore, bool sample, double corruptLevel,
			int gridSize, int iterations, string activation, string actParameter, bool trainAll, bool normalizeOutputs, double boost)
		{
			if (model.Equals("baseline"))
			{
				return new BaselineLearner();
			}
			else if (model.Equals("perceptron"))
			{
				return new Perceptron(rand, rate);
			}
			else if (model.Equals("neuralnet"))
			{
				return new NeuralNet(rand, rate, momentum, activation, actParameter, hidden, normalizeOutputs);
			}
			else if (model.Equals("decisiontree"))
			{
				return new DecisionTree(prune);
			}
			else if (model.Equals("knn"))
			{
				return new InstanceBasedLearner(distance, k, prune);
			}
			else if (model.Equals("clustering"))
			{
				return new Clustering(learnParameter, k, rand, ignore);
			}
			else if (model.Equals("dropout"))
			{
				return new Dropout(rand, rate, momentum, ph, pi, hidden);
			}
			else if (model.Equals("bptt"))
			{
				if ((hidden != null) && (hidden.Length > 1))
				{
					throw new Exception("Only one hidden layer is supported");
				}
				return new BPTT(rand, rate, momentum, k, hidden[0]);
			}
			else if (model.Equals("dbn"))
			{
				return new DBN(rand, rate, momentum, ph, pi, sample, hidden);
			}
			else if (model.Equals("sae"))
			{
				return new SAE(rand, rate, momentum, corruptLevel, hidden);
			}
			else if (model.Equals("relu"))
			{
				return new ReLU(rand, rate, momentum, hidden);
			}
			else if (model.Equals("som"))
			{
				return new SOM(rand, rate, iterations, gridSize);
			}
			else if (model.Equals("smlp"))
			{
				return new SMLP(rand, rate, momentum, corruptLevel, activation, actParameter, trainAll, hidden);
			}
			else if (model.Equals("amlp"))
			{
				return new AMLP(rand, rate, momentum, activation, actParameter, hidden);
			}
			else if (model.Equals("dmlp"))
			{
				return new DMLP(rand, rate, momentum, activation, actParameter, hidden);
			}
			else if (model.Equals("rnn"))
			{
				return new RNN(rand, rate, momentum, activation, actParameter, hidden, boost);
			}
			else if (model.Equals("nnrev"))
			{
				return new NNRev(rand, rate, momentum, activation, actParameter, hidden);
			}
			else
			{
				throw new Exception("Unrecognized model: " + model);
			}
		}

		public void Run(string[] args)
		{
			//args = new string[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

			//Random rand = new Random(1234); // Use a seed for deterministic results (makes debugging easier)
			Random rand = new Random(); // No seed for non-deterministic results

			//Parse the command line arguments
			ArgParser parser = new ArgParser(args);
			string fileName = parser.GetARFF(); //File specified by the user
			string learnerName = parser.GetLearner(); //Learning algorithm specified by the user
			string learnParameter = parser.GetLearnParameter(); // Learning parameter specified by the user
			string evalMethod = parser.GetEvaluation(); //Evaluation method specified by the user
			string evalParameter = parser.GetEvalParameter(); //Evaluation parameters specified by the user
			double rate = parser.GetRate();
			double momentum = parser.GetMomentum();
			double ph = parser.GetPH();
			double pi = parser.GetPI();
			int[] hidden = parser.GetHidden();
			int outputs = parser.GetOutputs();
			string outputFileName = parser.GetOutputFileName();
			string weightsFileName = parser.GetWeightsFileName();
			bool verbose = parser.GetVerbose();
			bool normalize = parser.GetNormalize();
			bool normalizeOutputs = parser.GetNormalizeOutputs();
			bool prune = parser.GetPrune();
			bool distance = parser.GetDistance();
			int k = parser.GetK();
			string ignore = parser.GetIgnore();
			bool sample = parser.GetSample();
			double corruptLevel = parser.GetCorruptLevel();
			int gridSize = parser.GetGridSize();
			int iterations = parser.GetIterations();
			string activation = parser.GetActivation();
			string actParameter = parser.GetActParameter();
			bool trainAll = parser.GetTrainAll();
			double boost = parser.GetBoost();

			// Load the model
			SupervisedLearner learner = GetLearner(learnerName, learnParameter, rand, rate, momentum, ph, pi, hidden, prune, distance, k, ignore, sample, corruptLevel, gridSize, iterations, activation, actParameter, trainAll, normalizeOutputs, boost);

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
			Matrix data = new Matrix();
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
				using (StreamWriter w = new StreamWriter(learner.OutputFileName))
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
				VMatrix features = new VMatrix(data, 0, 0, data.Rows(), data.Cols() - outputs);
				VMatrix labels = new VMatrix();
				if (outputs > 0)
				{
					labels = new VMatrix(data, 0, data.Cols() - outputs, data.Rows(), outputs);
				}
				Matrix confusion = new Matrix();
				long startTime = DateTime.Now.Ticks;
				double[] colMin = new double[data.Cols()];
				double[] colMax = new double[data.Cols()];
				for (var col = 0; col < data.Cols(); col++)
				{
					colMin[col] = data.ColumnMinOrig(col);
					colMax[col] = data.ColumnMaxOrig(col);
				}
				learner.VTrain(features, labels, colMin, colMax);
				TimeSpan elapsedTime = new TimeSpan(DateTime.Now.Ticks - startTime);
				Console.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
				if (outputs > 0)
				{
					double accuracy = learner.VMeasureAccuracy(features, labels, confusion);
					Console.WriteLine("Training set accuracy: " + accuracy);

					if (!string.IsNullOrEmpty(learner.OutputFileName))
					{
						using (StreamWriter w = File.AppendText(learner.OutputFileName))
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
				Matrix testData = new Matrix();
				testData.LoadArff(evalParameter);
				if (normalize)
				{
					testData.Normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
				}

				Console.WriteLine("Calculating accuracy on separate test set...");
				Console.WriteLine("Test set name: " + evalParameter);
				Console.WriteLine("Number of test instances: " + testData.Rows());
				VMatrix features = new VMatrix(data, 0, 0, data.Rows(), data.Cols() - outputs);
				VMatrix labels = new VMatrix(data, 0, data.Cols() - outputs, data.Rows(), outputs);
				long startTime = DateTime.Now.Ticks;
				double[] colMin = new double[data.Cols()];
				double[] colMax = new double[data.Cols()];
				for (var col = 0; col < data.Cols(); col++)
				{
					colMin[col] = data.ColumnMinOrig(col);
					colMax[col] = data.ColumnMaxOrig(col);
				}
				learner.VTrain(features, labels, colMin, colMax);
				TimeSpan elapsedTime = new TimeSpan(DateTime.Now.Ticks - startTime);
				Console.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
				double trainAccuracy = learner.VMeasureAccuracy(features, labels, null);
				Console.WriteLine("Training set accuracy: " + trainAccuracy);
				VMatrix testFeatures = new VMatrix(testData, 0, 0, testData.Rows(), testData.Cols() - outputs);
				VMatrix testLabels = new VMatrix(testData, 0, testData.Cols() - outputs, testData.Rows(), outputs);
				Matrix confusion = new Matrix();
				double testAccuracy = learner.VMeasureAccuracy(testFeatures, testLabels, confusion);
				Console.WriteLine("Test set accuracy: " + testAccuracy);
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (StreamWriter w = File.AppendText(learner.OutputFileName))
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
				double trainPercent = Double.Parse(evalParameter);
				if (trainPercent < 0 || trainPercent > 1)
				{
					throw new Exception("Percentage for random evaluation must be between 0 and 1");
				}
				Console.WriteLine("Percentage used for training: " + trainPercent);
				Console.WriteLine("Percentage used for testing: " + (1 - trainPercent));
				VMatrix vData = new VMatrix(data, 0, 0, data.Rows(), data.Cols());
				if (!(learner is BPTT))
				{
					vData.Shuffle(rand);
				}
				int trainSize = (int)(trainPercent * vData.Rows());
				VMatrix trainFeatures = new VMatrix(vData, 0, 0, trainSize, vData.Cols() - outputs);
				VMatrix trainLabels = new VMatrix(vData, 0, vData.Cols() - outputs, trainSize, outputs);
				VMatrix testFeatures = new VMatrix(vData, trainSize, 0, vData.Rows() - trainSize, vData.Cols() - outputs);
				VMatrix testLabels = new VMatrix(vData, trainSize, vData.Cols() - outputs, vData.Rows() - trainSize, outputs);
				long startTime = DateTime.Now.Ticks;
				double[] colMin = new double[data.Cols()];
				double[] colMax = new double[data.Cols()];
				for (var col = 0; col < data.Cols(); col++)
				{
					colMin[col] = vData.ColumnMinOrig(col);
					colMax[col] = data.ColumnMaxOrig(col);
				}
				learner.VTrain(trainFeatures, trainLabels, colMin, colMax);
				TimeSpan elapsedTime = new TimeSpan(DateTime.Now.Ticks - startTime);
				Console.WriteLine("Time to train (in seconds): " + elapsedTime.TotalSeconds);
				double trainAccuracy = learner.VMeasureAccuracy(trainFeatures, trainLabels, null);
				Console.WriteLine("Training set accuracy: " + trainAccuracy);
				Matrix confusion = new Matrix();
				double testAccuracy = learner.VMeasureAccuracy(testFeatures, testLabels, confusion);
				Console.WriteLine("Test set accuracy: " + testAccuracy);
				double testMSE = learner.VGetMSE(testFeatures, testLabels);
				Console.WriteLine("Test set MSE: " + testMSE);
                
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (StreamWriter w = File.AppendText(learner.OutputFileName))
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
				int folds = int.Parse(evalParameter);
				if (folds <= 0)
				{
					throw new Exception("Number of folds must be greater than 0");
				}
				Console.WriteLine("Number of folds: " + folds);
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (StreamWriter w = File.AppendText(learner.OutputFileName))
					{
						w.WriteLine();
						w.WriteLine("Calculating accuracy using cross-validation...");
						w.WriteLine("Number of folds: " + folds);
					}
				}
				int reps = 1;
				double sumAccuracy = 0.0;
				long ticks = 0;
				for (int j = 0; j < reps; j++)
				{
					data.Shuffle(rand);
					for (int i = 0; i < folds; i++)
					{
						int begin = i * data.Rows() / folds;
						int end = (i + 1) * data.Rows() / folds;
						Matrix trainFeatures = new Matrix(data, 0, 0, begin, data.Cols() - outputs);
						Matrix trainLabels = new Matrix(data, 0, data.Cols() - outputs, begin, outputs);
						Matrix testFeatures = new Matrix(data, begin, 0, end - begin, data.Cols() - outputs);
						Matrix testLabels = new Matrix(data, begin, data.Cols() - outputs, end - begin, outputs);
						trainFeatures.Add(data, end, 0, data.Rows() - end);
						trainLabels.Add(data, end, data.Cols() - outputs, data.Rows() - end);
						long startTime = DateTime.Now.Ticks;
						double[] colMin = new double[data.Cols()];
						double[] colMax = new double[data.Cols()];
						for (var col = 0; col < data.Cols(); col++)
						{
							colMin[col] = data.ColumnMinOrig(col);
							colMax[col] = data.ColumnMaxOrig(col);
						}
						learner.Train(trainFeatures, trainLabels, colMin, colMax);
						ticks = DateTime.Now.Ticks - startTime;
						double accuracy = learner.MeasureAccuracy(testFeatures, testLabels, null);
						sumAccuracy += accuracy;
						Console.WriteLine("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
						if (!string.IsNullOrEmpty(learner.OutputFileName))
						{
							using (StreamWriter w = File.AppendText(learner.OutputFileName))
							{
								w.WriteLine("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
								w.WriteLine();
							}
						}
					}
				}
				ticks /= (reps * folds);
				TimeSpan elapsedTime = new TimeSpan(ticks);
				Console.WriteLine("Average time to train (in seconds): " + elapsedTime.TotalSeconds);
				Console.WriteLine("Mean accuracy=" + (sumAccuracy / (reps * folds)));
				if (!string.IsNullOrEmpty(learner.OutputFileName))
				{
					using (StreamWriter w = File.AppendText(learner.OutputFileName))
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