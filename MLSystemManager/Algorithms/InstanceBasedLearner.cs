using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	public class InstanceBasedLearner : SupervisedLearner
	{
		private bool Distance { get; set; }
		private int K { get; set; }
		private bool Prune { get; set; }
		private Matrix Features { get; set; }
		private Matrix Labels { get; set; }
		private double[] Distances { get; set; }

		public InstanceBasedLearner()
		{
		}

		public InstanceBasedLearner(Parameters parameters)
		{
			Distance = parameters.Distance;
			K = parameters.K;
			Prune = parameters.Prune;
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
			Features = new Matrix(features, 0, 0, features.Rows(), features.Cols());
			Labels = new Matrix(labels, 0, 0, labels.Rows(), labels.Cols());
			Distances = new double[Features.Rows()];

			if (!string.IsNullOrEmpty(OutputFileName))
			{
				using (StreamWriter w = File.AppendText(OutputFileName))
				{
					w.WriteLine();
					w.WriteLine("Distance: " + Distance);
					w.WriteLine("K: " + K);
					w.WriteLine("Prune: " + Prune);
				}
			}

			if (Prune)
			{
				DoPrune();
			}
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
		}

		private void DoPrune()
		{
			bool isContinuous = (Labels.ValueCount(0) < 2);

			for (var row = 0; row < Features.Rows(); row++)
			{
				int[] nearest = FindKnn(Features.Row(row), new int[1] { row });
				int with = 0;
				double withSSE = 0;
				double[] pred = new double[1];

				for (var n = 0; n < nearest.Length; n++)
				{
					pred[0] = 0;
					double[] targ = Labels.Row(nearest[n]);
					GetOutput(FindKnn(Features.Row(nearest[n]), new int[1] { nearest[n] }), pred);
					if (isContinuous)
					{
						double delta = targ[0] - pred[0];
						withSSE += (delta * delta);
					}
					else if (pred[0] == targ[0])
					{
						with++;
					}
				}

				int without = 0;
				double withoutSSE = 0;
				for (var n = 0; n < nearest.Length; n++)
				{
					pred[0] = 0;
					double[] targ = Labels.Row(nearest[n]);
					GetOutput(FindKnn(Features.Row(nearest[n]), new int[2] { row, nearest[n] }), pred);
					if (isContinuous)
					{
						double delta = targ[0] - pred[0];
						withoutSSE += (delta * delta);
					}
					else if (pred[0] == targ[0])
					{
						without++;
					}
				}


				bool remove = false;

				if (isContinuous)
				{
					withSSE = Math.Sqrt(withSSE / K);
					withoutSSE = Math.Sqrt(withoutSSE / K);
					if (Math.Abs(withSSE - withoutSSE) <= (0.1 * withSSE))
					{
						remove = true;
					}
				}
				else if (without - with >= 0)
				{
					remove = true;
				}

				if (remove)
				{
					// remove the row
					Features.Delete(row);
					Labels.Delete(row);
					row--;
				}
			}

			if (!string.IsNullOrEmpty(OutputFileName))
			{
				using (StreamWriter w = File.AppendText(OutputFileName))
				{
					w.WriteLine();
					w.WriteLine("After pruning:");
					w.WriteLine("Number of instances: " + Features.Rows());
				}
			}
		}

		/// <summary>
		/// Find the k-nn for the specified feature
		/// </summary>
		/// <param name="features">array of feature values</param>
		/// <param name="ignore">array of rows to ignore</param>
		/// <returns></returns>
		private int[] FindKnn(double[] features, int[] ignore = null)
		{
			int[] nearest = new int[K];

			// initialize the nearest
			for (var i = 0; i < nearest.Length; i++)
			{
				nearest[i] = -1;
			}

			// find the distances to the training features
			for (var row = 0; row < Features.Rows(); row++)
			{
				// skip any rows in the ignore array
				if ((ignore != null) && ignore.Contains(row))
				{
					continue;
				}

				Distances[row] = GetMinkowskyDistance(features, row);

				// see if this distance is one of the k nearest
				for (var i = 0; i < K; i++)
				{
					if ((nearest[i] < 0) || (Distances[row] < Distances[nearest[i]]))
					{
						if (nearest[i] >= 0)
						{
							// make room
							for (var j = K - 1; j > i; j--)
							{
								nearest[j] = nearest[j - 1];
							}
						}

						nearest[i] = row;
						break;
					}
				}
			}

			return nearest;
		}

		private double GetDistance(double[] features, int row)
		{
			double distance = 0;

			for (var col = 0; col < Features.Cols(); col++)
			{
				if (Features.ValueCount(col) < 2)
				{
					// continuous
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else
					{
						double d = Math.Pow(features[col] - Features.Get(row, col), 2);
						distance += d;
					}
				}
				else
				{
					// nominal
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else if (features[col] != Features.Get(row, col))
					{
						distance += 1;
					}
				}
			}

			distance = Math.Sqrt(distance);

			return distance;
		}

		private double GetCanberraDistance(double[] features, int row)
		{
			double distance = 0;

			for (var col = 0; col < Features.Cols(); col++)
			{
				if (Features.ValueCount(col) < 2)
				{
					// continuous
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else
					{
						double d = Math.Abs(features[col] - Features.Get(row, col)) / (Math.Abs(features[col]) + Math.Abs(Features.Get(row, col)));
						distance += d;
					}
				}
				else
				{
					// nominal
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else if (features[col] != Features.Get(row, col))
					{
						distance += 1;
					}
				}
			}

			return distance;
		}

		private double GetManhattanDistance(double[] features, int row)
		{
			double distance = 0;

			for (var col = 0; col < Features.Cols(); col++)
			{
				if (Features.ValueCount(col) < 2)
				{
					// continuous
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else
					{
						double d = Math.Abs(features[col] - Features.Get(row, col));
						distance += d;
					}
				}
				else
				{
					// nominal
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else if (features[col] != Features.Get(row, col))
					{
						distance += 1;
					}
				}
			}

			return distance;
		}

		private double GetMinkowskyDistance(double[] features, int row)
		{
			double distance = 0;
			double r = 3.0;

			for (var col = 0; col < Features.Cols(); col++)
			{
				if (Features.ValueCount(col) < 2)
				{
					// continuous
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else
					{
						double d = Math.Pow(Math.Abs(features[col] - Features.Get(row, col)), r);
						distance += d;
					}
				}
				else
				{
					// nominal
					if ((features[col] == Matrix.MISSING) || (Features.Get(row, col) == Matrix.MISSING))
					{
						distance += 1;
					}
					else if (features[col] != Features.Get(row, col))
					{
						distance += 1;
					}
				}
			}

			distance = Math.Pow(distance, 1.0 / r);
			return distance;
		}

		private int GetOutput(int[] nearest, double[] labels)
		{
			double sumWeights = 0;
			bool isContinuous = Labels.ValueCount(0) < 2;
			double[] output = new double[isContinuous ? 1 : Labels.ValueCount(0)];
			int[] count = new int[output.Length];
			int result = 0;

			// calculate the output
			for (var k = 0; k < nearest.Length; k++)
			{
				if (nearest[k] >= 0)
				{
					double weight = 1.0;
					if (Distance && (Distances[nearest[k]] > 0))
					{
						weight /= Math.Pow(Distances[nearest[k]], 2);
					}
					sumWeights += weight;

					if (isContinuous)
					{
						// continuous
						output[0] += weight * Labels.Get(nearest[k], 0);
						count[0]++;
					}
					else
					{
						int idx = (int)Labels.Get(nearest[k], 0);
						output[idx] += weight;
						count[idx]++;
					}
				}
			}

			if (isContinuous)
			{
				labels[0] = output[0] / sumWeights;
			}
			else
			{
				double max = double.MinValue;
				for (var c = 0; c < output.Length; c++)
				{
					if (output[c] > max)
					{
						labels[0] = c;
						max = output[c];
						result = count[c];
					}
				}
			}

			return result;
		}

		public override void Predict(double[] features, double[] labels)
		{
			int[] nearest = FindKnn(features);
			GetOutput(nearest, labels);
		}
	}
}
