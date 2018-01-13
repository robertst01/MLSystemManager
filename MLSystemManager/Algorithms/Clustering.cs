using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	class Clustering : SupervisedLearner
	{
		private string _algorithm;
		private int _k;
		private Random _rand;
		private VMatrix _features;
		private VMatrix _labels;
		private List<int> _ignore;
		private List<Cluster> _clusters;
		private double[,] _distances;
		private StreamWriter _outputFile;

		private class Cluster
		{
			public int Number { get; set; }
			private VMatrix Features { get; set; }
			private List<int> Ignore { get; set; }
			public List<int> Instances { get; set; }
			public double[] Centroid { get; set; }

			public Cluster(int number, VMatrix features, int row, List<int> ignore)
			{
				Number = number;
				Features = features;
				Centroid = new double[features.Cols()];
				for (var col = 0; col < Centroid.Length; col++)
				{
					Centroid[col] = features.Get(row, col);
				}

				Instances = new List<int>();
				Ignore = ignore;
			}

			public void AddInstance(int row)
			{
				Instances.Add(row);
			}

			public double GetError(double[] features)
			{
				double error = 0;

				for (var col = 0; col < Features.Cols(); col++)
				{
					if (!Ignore.Contains(col))
					{
						if (Features.ValueCount(col) < 2)
						{
							// continuous
							if ((Centroid[col] == Matrix.MISSING) || (features[col] == Matrix.MISSING))
							{
								error += 1;
							}
							else
							{
								error += Math.Pow(Centroid[col] - features[col], 2);
							}
						}
						else
						{
							// nominal
							if ((Centroid[col] == Matrix.MISSING) || (features[col] == Matrix.MISSING))
							{
								error += 1;
							}
							else if (Centroid[col] != features[col])
							{
								error += 1;
							}
						}
					}
				}

				return error;
			}

			public double GetDistance(double[] features)
			{
				return Math.Sqrt(GetError(features));
			}

			public double GetSSE()
			{
				double sse = 0;

				foreach (var i in Instances)
				{
					sse += GetError(Features.Row(i));
				}

				return sse;
			}

			public void Recalculate()
			{
				// recalculate the centroid
				for (var col = 0; col < Centroid.Length; col++)
				{
					if (!Ignore.Contains(col))
					{
						Centroid[col] = GetValue(col);
					}
				}
			}

			private double GetValue(int col)
			{
				bool isContinuous = Features.ValueCount(col) < 2;
				int[] count = new int[isContinuous ? 1 : Features.ValueCount(col)];
				double result = 0;

				// calculate the output
				for (var i = 0; i < Instances.Count; i++)
				{
					if (Features.Get(Instances[i], col) != Matrix.MISSING)
					{
						if (isContinuous)
						{
							// continuous
							result += Features.Get(Instances[i], col);
							count[0]++;
						}
						else
						{
							int idx = (int)Features.Get(Instances[i], col);
							count[idx]++;
						}
					}
				}

				if (isContinuous)
				{
					if (count[0] > 0)
					{
						result /= count[0];
					}
					else
					{
						result = Matrix.MISSING;
					}
				}
				else
				{
					double max = count[0];
					result = 0;
					for (var c = 1; c < count.Length; c++)
					{
						if (count[c] > max)
						{
							result = c;
							max = count[c];
						}
					}

					if (max == 0)
					{
						result = Matrix.MISSING;
					}
				}

				return result;
			}

			public void PrintCentroid(StreamWriter outputFile)
			{
				outputFile.Write(string.Format("Centroid {0} = ", Number));
				int count = 0;
				for (var col = 0; col < Features.Cols(); col++)
				{
					if (!Ignore.Contains(col))
					{
						if (count++ > 0)
						{
							outputFile.Write(", ");
						}

						if (Centroid[col] == Matrix.MISSING)
						{
							outputFile.Write("?");
						}
						else if (Features.ValueCount(col) < 2)
						{
							// continuous
							outputFile.Write(Centroid[col]);
						}
						else
						{
							// nominal
							outputFile.Write(Features.AttrValue(col, (int)Centroid[col]));
						}
					}
				}

				outputFile.WriteLine();
			}

			public void PrintInstances(StreamWriter outputFile)
			{
				outputFile.Write("Cluster " + Number + ":");
				foreach (var i in Instances)
				{
					outputFile.Write(" " + i);
				}
				outputFile.WriteLine();
			}

			public void ClearInstances()
			{
				Instances.Clear();
			}
		}

		public Clustering()
		{
		}

		public Clustering(Parameters parameters)
		{
			_algorithm = parameters.LearnExtra;
			_k = parameters.K;
			_rand = Rand.Get();

			_ignore = new List<int>();
			if (!string.IsNullOrEmpty(parameters.Ignore))
			{
				var si = parameters.Ignore.Substring(1, parameters.Ignore.Length - 2).Split(',');
				foreach (var s in si)
				{
					_ignore.Add(int.Parse(s));
				}
			}
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			_features = new VMatrix(features, 0, 0, features.Rows(), features.Cols());
			if (labels.Data != null)
			{
				_labels = new VMatrix(labels, 0, 0, labels.Rows(), labels.Cols());
			}
			_clusters = new List<Cluster>();
			if (!string.IsNullOrEmpty(OutputFileName))
			{
				_outputFile = File.AppendText(OutputFileName);
				_outputFile.Write("Algorithm: ");
			}

			if (_algorithm == "k")
			{
				if (_outputFile != null)
				{
					_outputFile.WriteLine("k-means (k = " + _k + ")");
				}

//				Features.Shuffle(Rand, Labels);

				// create the initial clusters
				for (var k = 0; k < _k; k++)
				{
					var cluster = new Cluster(k, _features, k, _ignore);
					_clusters.Add(cluster);
					if (_outputFile != null)
					{
						cluster.PrintCentroid(_outputFile);
					}
				}

				double lastSsd = double.MinValue;

				for (;;)
				{
					var ssd = TrainK();
					if (_outputFile != null)
					{
						_outputFile.WriteLine(string.Format("Sum squared-distance of each row with its centroid={0}", ssd));
					}

					if (ssd != lastSsd)
					{
						lastSsd = ssd;
						if (_outputFile != null)
						{
							_outputFile.WriteLine("Recomputing the centroids of each cluster...");
						}
						foreach (var cluster in _clusters)
						{
							cluster.Recalculate();
							cluster.ClearInstances();
							if (_outputFile != null)
							{
								cluster.PrintCentroid(_outputFile);
							}
						}
					}
					else
					{
						break;
					}
				}
			}
			else if (_algorithm == "single")
			{
				if (_outputFile != null)
				{
					_outputFile.WriteLine("HAC single (k = " + _k + ")");
				}

				// create the initial clusters
				for (var row = 0; row < _features.Rows(); row++)
				{
					var cluster = new Cluster(0, _features, row, _ignore);
					cluster.AddInstance(row);
					_clusters.Add(cluster);
				}

				// create the distance matrix
				_distances = new double[_features.Rows(), _features.Rows()];

				for (var row = 0; row < _features.Rows(); row++)
				{
					for (var row2 = row; row2 < _features.Rows(); row2++)
					{
						double distance = 0;
						if (row2 > row)
						{
							distance = _clusters[row].GetDistance(_features.Row(row2));
						}
						_distances[row, row2] = distance;
						if (row != row2)
						{
							_distances[row2, row] = distance;
						}
					}
				}

				int iteration = 0;

				do
				{
					TrainSingle(iteration++);
				} while (_clusters.Count > _k);
			}
			else if (_algorithm == "complete")
			{
				if (_outputFile != null)
				{
					_outputFile.WriteLine("HAC complete (k = " + _k + ")");
				}

				// create the initial clusters
				for (var row = 0; row < _features.Rows(); row++)
				{
					var cluster = new Cluster(0, _features, row, _ignore);
					cluster.AddInstance(row);
					_clusters.Add(cluster);
				}

				// create the distance matrix
				_distances = new double[_features.Rows(), _features.Rows()];

				for (var row = 0; row < _features.Rows(); row++)
				{
					for (var row2 = row; row2 < _features.Rows(); row2++)
					{
						double distance = 0;
						if (row2 > row)
						{
							distance = _clusters[row].GetDistance(_features.Row(row2));
						}
						_distances[row, row2] = distance;
						if (row != row2)
						{
							_distances[row2, row] = distance;
						}
					}
				}

				int iteration = 0;

				do
				{
					TrainComplete(iteration++);
				} while (_clusters.Count > _k);
			}
			else if (_algorithm == "average")
			{
				if (_outputFile != null)
				{
					_outputFile.WriteLine("HAC average (k = " + _k + ")");
				}

				// create the initial clusters
				for (var row = 0; row < _features.Rows(); row++)
				{
					var cluster = new Cluster(0, _features, row, _ignore);
					cluster.AddInstance(row);
					_clusters.Add(cluster);
				}

				// create the distance matrix
				_distances = new double[_features.Rows(), _features.Rows()];

				for (var row = 0; row < _features.Rows(); row++)
				{
					for (var row2 = row; row2 < _features.Rows(); row2++)
					{
						double distance = 0;
						if (row2 > row)
						{
							distance = _clusters[row].GetDistance(_features.Row(row2));
						}
						_distances[row, row2] = distance;
						if (row != row2)
						{
							_distances[row2, row] = distance;
						}
					}
				}

				int iteration = 0;

				do
				{
					TrainAverage(iteration++);
				} while (_clusters.Count > _k);
			}
			else
			{
				throw new Exception("Inavlid Algorithm - " + _algorithm);
			}

			if (_outputFile != null)
			{
				_outputFile.WriteLine();
				_outputFile.WriteLine("Cluster centroids:");
				
				_outputFile.Write("Cluster#\t\t\t");
				for (var c = 0; c < _clusters.Count; c++)
				{
					_outputFile.Write("\t\t" + c);
				}
				_outputFile.WriteLine();

				_outputFile.Write("# of instances:\t\t\t");
				for (var c = 0; c < _clusters.Count; c++)
				{
					_outputFile.Write("\t\t" + _clusters[c].Instances.Count);
				}
				_outputFile.WriteLine();

				_outputFile.WriteLine("==========================================================================================================");
				for (var col = 0; col < _features.Cols(); col++)
				{
					if (!_ignore.Contains(col))
					{
						_outputFile.Write(_features.AttrName(col));
						foreach (var cluster in _clusters)
						{
							if (cluster.Centroid[col] == Matrix.MISSING)
							{
								_outputFile.Write("\t?");
							}
							else if (_features.ValueCount(col) < 2)
							{
								// continuous
								_outputFile.Write(string.Format("\t{0:0.#####}", cluster.Centroid[col]));
							}
							else
							{
								_outputFile.Write("\t" + _features.AttrValue(col, (int)cluster.Centroid[col]));
							}
						}
						_outputFile.WriteLine();
					}
				}

				double sse = 0;
				_outputFile.Write("Sum squared error:\t");
				foreach (var cluster in _clusters)
				{
					var error = cluster.GetSSE();
					sse += error;
					_outputFile.Write(string.Format("\t{0:0.#####}", error));
				}
				_outputFile.WriteLine();

				_outputFile.WriteLine("Number of clusters: " + _clusters.Count);
				_outputFile.WriteLine(string.Format("Total sum squared error: {0:0.#####}", sse));
				_outputFile.WriteLine(string.Format("DBI: {0}", GetDBI()));
			}

			if (_outputFile != null)
			{
				_outputFile.Close();
			}
		}

		private double TrainK()
		{
			if (_outputFile != null)
			{
				_outputFile.WriteLine("Assigning each row to the cluster of the nearest centroid...");
				_outputFile.WriteLine("The cluster assignments are:");
			}

			// add the training set elements to the clusters
			for (var row = 0; row < _features.Rows(); row++)
			{
				var cluster = GetNearestCluster(_features.Row(row));
				cluster.AddInstance(row);

				if (_outputFile != null)
				{
					if (row % 10 == 0)
					{
						_outputFile.WriteLine();
						_outputFile.Write("\t");
					}
					else
					{
						_outputFile.Write(", ");
					}
					_outputFile.Write(string.Format("{0}={1}", row, cluster.Number));
				}
			}

			if (_outputFile != null)
			{
				_outputFile.WriteLine();
			}

			double sse = 0;

			foreach (var cluster in _clusters)
			{
				sse += cluster.GetSSE();
			}

			return sse;
		}

		private Cluster GetNearestCluster(double[] features)
		{
			Cluster nearest = _clusters[0];
			double nDistance = double.MaxValue;

			foreach (var cluster in _clusters)
			{
				var distance = cluster.GetDistance(features);
				if (distance < nDistance)
				{
					nearest = cluster;
					nDistance = distance;
				}
			}

			return nearest;
		}

		private double TrainSingle(int iteration)
		{
			double minDist = double.MaxValue;
			int cluster1 = 0;
			int cluster2 = 0;

			if (_outputFile != null)
			{
				_outputFile.WriteLine();
				_outputFile.WriteLine("--------------");
				_outputFile.WriteLine("Iteration " + iteration);
				_outputFile.WriteLine("--------------");
			}

			// find the nearest clusters
			for (var beg = 0; beg < _clusters.Count - 1; beg++)
			{
				for (var end = beg + 1; end < _clusters.Count; end++)
				{
					foreach (var begi in _clusters[beg].Instances)
					{
						foreach (var endi in _clusters[end].Instances)
						{
							if (_distances[begi, endi] < minDist)
							{
								minDist = _distances[begi, endi];
								cluster1 = beg;
								cluster2 = end;
							}
						}
					}
				}
			}

			if (_outputFile != null)
			{
				_outputFile.WriteLine("Merging clusters {0} and {1}\tDistance: {2}", cluster1, cluster2, minDist);
				_outputFile.WriteLine();
			}

			// merge the clusters
			foreach (var i in _clusters[cluster2].Instances)
			{
				_clusters[cluster1].AddInstance(i);
			}

			_clusters[cluster1].Recalculate();
			_clusters.RemoveAt(cluster2);

			for (var c = 0; c < _clusters.Count; c++)
			{
				_clusters[c].Number = c;
				_clusters[c].PrintInstances(_outputFile);
			}

			double sse = 0;

			foreach (var cluster in _clusters)
			{
				sse += cluster.GetSSE();
			}

			return sse;
		}

		private double TrainComplete(int iteration)
		{
			double minDist = double.MaxValue;
			int cluster1 = 0;
			int cluster2 = 0;

			if (_outputFile != null)
			{
				_outputFile.WriteLine();
				_outputFile.WriteLine("--------------");
				_outputFile.WriteLine("Iteration " + iteration);
				_outputFile.WriteLine("--------------");
			}

			// find the nearest clusters
			for (var beg = 0; beg < _clusters.Count - 1; beg++)
			{
				for (var end = beg + 1; end < _clusters.Count; end++)
				{
					double maxDist = double.MinValue;
					foreach (var begi in _clusters[beg].Instances)
					{
						foreach (var endi in _clusters[end].Instances)
						{
							if (_distances[begi, endi] > maxDist)
							{
								maxDist = _distances[begi, endi];
							}
						}
					}
					if (maxDist < minDist)
					{
						minDist = maxDist;
						cluster1 = beg;
						cluster2 = end;
					}
				}
			}

			if (_outputFile != null)
			{
				_outputFile.WriteLine("Merging clusters {0} and {1}\tDistance: {2}", cluster1, cluster2, minDist);
				_outputFile.WriteLine();
			}

			// merge the clusters
			foreach (var i in _clusters[cluster2].Instances)
			{
				_clusters[cluster1].AddInstance(i);
			}

			_clusters[cluster1].Recalculate();
			_clusters.RemoveAt(cluster2);

			for (var c = 0; c < _clusters.Count; c++)
			{
				_clusters[c].Number = c;
				_clusters[c].PrintInstances(_outputFile);
			}

			double sse = 0;

			foreach (var cluster in _clusters)
			{
				sse += cluster.GetSSE();
			}

			return sse;
		}

		private double TrainAverage(int iteration)
		{
			double minDist = double.MaxValue;
			int cluster1 = 0;
			int cluster2 = 0;

			if (_outputFile != null)
			{
				_outputFile.WriteLine();
				_outputFile.WriteLine("--------------");
				_outputFile.WriteLine("Iteration " + iteration);
				_outputFile.WriteLine("--------------");
			}

			// find the nearest clusters
			for (var beg = 0; beg < _clusters.Count - 1; beg++)
			{
				for (var end = beg + 1; end < _clusters.Count; end++)
				{
					double totDist = 0;
					int count = 0;
					foreach (var begi in _clusters[beg].Instances)
					{
						foreach (var endi in _clusters[end].Instances)
						{
							totDist += _distances[begi, endi];
							count++;
						}
					}

					totDist /= count;
					if (totDist < minDist)
					{
						minDist = totDist;
						cluster1 = beg;
						cluster2 = end;
					}
				}
			}

			if (_outputFile != null)
			{
				_outputFile.WriteLine("Merging clusters {0} and {1}\tDistance: {2}", cluster1, cluster2, minDist);
				_outputFile.WriteLine();
			}

			// merge the clusters
			foreach (var i in _clusters[cluster2].Instances)
			{
				_clusters[cluster1].AddInstance(i);
			}

			_clusters[cluster1].Recalculate();
			_clusters.RemoveAt(cluster2);

			for (var c = 0; c < _clusters.Count; c++)
			{
				_clusters[c].Number = c;
				_clusters[c].PrintInstances(_outputFile);
			}

			double sse = 0;

			foreach (var cluster in _clusters)
			{
				sse += cluster.GetSSE();
			}

			return sse;
		}

		private double GetDBI()
		{
			double dbi = 0;

			for (var c1 = 0; c1 < _clusters.Count - 1; c1++)
			{
				double S1 = _clusters[c1].GetSSE() / _clusters[c1].Instances.Count;
				double maxR = double.MinValue;
				for (var c2 = c1 + 1; c2 < _clusters.Count; c2++)
				{
					double S2 = _clusters[c2].GetSSE() / _clusters[c2].Instances.Count;
					double d = _clusters[c1].GetError(_clusters[c2].Centroid);
					double R = (S1 + S2) / d;
					if (R > maxR)
					{
						maxR = R;
					}
				}

				dbi += maxR;
			}

			return dbi / _clusters.Count;
		}

		public override void Predict(double[] features, double[] labels)
		{
			var nearest = GetNearestCluster(features);
			GetOutput(nearest, labels);
		}

		private void GetOutput(Cluster cluster, double[] labels)
		{
			bool isContinuous = _labels.ValueCount(0) < 2;
			int[] count = new int[isContinuous ? 1 : _labels.ValueCount(0)];
			double result = 0;

			// calculate the output
			for (var i = 0; i < cluster.Instances.Count; i++)
			{
				if (isContinuous)
				{
					// continuous
					result += _labels.Get(cluster.Instances[i], 0);
					count[0]++;
				}
				else
				{
					double idx = _labels.Get(cluster.Instances[i], 0);
					if (idx != Matrix.MISSING)
					{
						count[(int)idx]++;
					}
				}
			}

			if (isContinuous)
			{
				result /= count[0];
			}
			else
			{
				double max = count[0];
				labels[0] = 0;
				for (var c = 1; c < count.Length; c++)
				{
					if (count[c] > max)
					{
						labels[0] = c;
						max = count[c];
					}
				}
			}
		}
	}
}
