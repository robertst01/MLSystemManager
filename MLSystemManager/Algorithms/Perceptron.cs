using System;
using System.Collections.Generic;
using System.IO;

namespace MLSystemManager.Algorithms
{
	public class Perceptron : SupervisedLearner
	{
		private List<double[]> m_weights;
		private Random m_rand;
		private double m_rate = 0.1;
		private StreamWriter m_outputFile = null;
		private int m_count = 1;
        
		public Perceptron()
		{
			m_rand = new Random();
		}

		public Perceptron(Parameters parameters)
		{
			m_rand = Rand.Get();
			m_rate = parameters.Rate;
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			if (labels.ValueCount(0) > 2)
			{
				m_count = labels.ValueCount(0);
			}

			// create one set of weights for each output
			m_weights = new List<double[]>();
			for (var p = 0; p < m_count; p++)
			{
				var weights = new double[features.Cols() + 1];
				for (var i = 0; i < weights.Length; i++)
				{
					weights[i] = 1.0 - (m_rand.NextDouble() * 2.0);
				}
				m_weights.Add(weights);
			}

			// iterate through each of the instances
			for (var instance = 0; instance < m_count; instance++)
			{
				double error;               // error rate for the current epoch
				double bestError = 1.0;     // best (smallest) error rate so far
				int eCount = 0;             // number of epochs since the best error
				int epoch = 0;              // current epoch number
				bool done = false;
				double bestAccuracy = 0;    // best accuracy so far
				int bestEpoch = 0;          // epoch number of best accuracy
				double[] bestWeights = new double[features.Cols() + 1];       // best weights

				if (m_outputFile != null)
				{
					m_outputFile.WriteLine("Instance " + instance);
					m_outputFile.WriteLine("Epoch\tError Rate");
				}

				do
				{
					// shuffle the training set
					features.Shuffle(m_rand, labels);

					error = TrainEpoch(instance, ++epoch, features, labels);

					// check the accuracy after this epoch
					double accuracy = GetAccuracy(instance, features, labels);
					if (accuracy > bestAccuracy)
					{
						// save the best for later
						bestAccuracy = accuracy;
						bestEpoch = epoch;
						for (int i = 0; i < bestWeights.Length; i++)
						{
							bestWeights[i] = m_weights[instance][i];
						}
					}

					if (error == 0.0)
					{
						// can't get better than this
						done = true;
					}
					else if ((epoch == 1) || (error <= bestError))
					{
						// save the best error so far
						bestError = error;
						eCount = 0;
					}
					else
					{
						// check to see if we're done
						eCount++;
						if (eCount >= 10)
						{
							done = true;
						}
					}
				} while (!done);

				if (m_outputFile != null)
				{
					m_outputFile.WriteLine();
					m_outputFile.WriteLine("Weights");
					for (var i = 0; i < m_weights[instance].Length - 1; i++)
					{
						m_outputFile.Write(string.Format("{0}\t", m_weights[instance][i]));
					}
					m_outputFile.WriteLine(string.Format("{0}\t", m_weights[instance][m_weights[instance].Length - 1]));
					m_outputFile.WriteLine();
				}

				if (bestEpoch != epoch)
				{
					for (int i = 0; i < bestWeights.Length; i++)
					{
						m_weights[instance][i] = bestWeights[i] ;
					}
					if (m_outputFile != null)
					{
						m_outputFile.WriteLine();
						m_outputFile.WriteLine(string.Format("Best Weights (from Epoch {0}, accuracy={1})", bestEpoch, bestAccuracy));
						for (var i = 0; i < m_weights[instance].Length - 1; i++)
						{
							m_outputFile.Write(string.Format("{0}\t", m_weights[instance][i]));
						}
						m_outputFile.WriteLine(string.Format("{0}\t", m_weights[instance][m_weights[instance].Length - 1]));
						m_outputFile.WriteLine();
					}
				}
			}

			if (m_outputFile != null)
			{
				m_outputFile.Close();
			}
		}

		private double TrainEpoch(int instance, int epoch, VMatrix features, VMatrix labels)
		{
			if (m_outputFile == null)
			{
				Console.WriteLine(epoch);
			}

			int eCount = 0;

			for (var row = 0; row < features.Rows(); row++)
			{
				double net = 0;

				// calculate the net value
				for (var col = 0; col < features.Cols(); col++)
				{
					net += m_weights[instance][col] * features.Row(row)[col];
				}

				// add the bias
				net += m_weights[instance][m_weights[instance].Length - 1];
                
				double z = (net > 0 ? 1.0 : 0);
				double t = labels.Row(row)[0];
				if (m_count > 2)
				{
					t = (t == instance) ? 1.0 : 0;
				}

				// check to see if the predicted matches the actual
				if (z != t)
				{
					eCount++;
					double delta;

					// adjust the weights
					for (var i = 0; i < m_weights[instance].Length - 1; i++)
					{
						delta = (t - z) * m_rate * features.Row(row)[i];
						//Console.Write(string.Format("{0}\t", delta));
						m_weights[instance][i] += delta;
					}
					// adjust the bias weight
					delta = (t - z) * m_rate;
					//Console.WriteLine(delta);
					m_weights[instance][m_weights[instance].Length - 1] += delta;
				}
			}

			// print the new weights
			if (m_outputFile == null)
			{
				for (var i = 0; i < m_weights[instance].Length - 1; i++)
				{
					Console.Write(string.Format("{0}\t", m_weights[instance][i]));
				}
				Console.WriteLine(m_weights[instance][m_weights[instance].Length - 1]);
			}

			double error = 1.0 * eCount / features.Rows();

			if (m_outputFile == null)
			{
				Console.WriteLine(error);
				Console.WriteLine();
			}
			else
			{
				m_outputFile.WriteLine(string.Format("{0}\t{1}", epoch, error));
			}

			return error;
		}

		// check the accuracy so far
		private double GetAccuracy(int instance, VMatrix features, VMatrix labels)
		{
			var eCount = 0;

			for (var row = 0; row < features.Rows(); row++)
			{
				double net = 0;

				for (var col = 0; col < features.Cols(); col++)
				{
					net += m_weights[instance][col] * features.Row(row)[col];
				}

					// add the bias
				net += m_weights[instance][m_weights[instance].Length - 1];

				double z = (net > 0 ? 1.0 : 0);
				double t = labels.Row(row)[0];
				if (m_count > 2)
				{
					t = (t == instance) ? 1.0 : 0;
				}

				if (t != z)
				{
					eCount++;
				}
			}

			return 1.0 - (1.0 * eCount / features.Rows());
		}

		public override void Predict(double[] features, double[] labels)
		{
			double[] net = new double[m_count];

			// calculate the net values
			for (var instance = 0; instance < m_count; instance++)
			{
				net[instance] = 0;

				for (var col = 0; col < features.Length; col++)
				{
					net[instance] += m_weights[instance][col] * features[col];
				}

				// add the bias
				net[instance] += m_weights[instance][m_weights[instance].Length - 1];
			}

			double z = (net[0] > 0 ? 1.0 : 0);

			// find the biggest
			if (m_count > 2)
			{
				z = 0;
				var max = net[0];

				for (var instance = 1; instance < m_count; instance++)
				{
					if (net[instance] > max)
					{
						z = instance;
						max = net[instance];
					}
				}
			}

			labels[0] = z;
		}
	}
}
