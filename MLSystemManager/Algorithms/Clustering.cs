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
		private string Algorithm { get; set; }
		private int K { get; set; }
		private Random Rand { get; set; }
		private VMatrix Features { get; set; }
		private VMatrix Labels { get; set; }
		private List<int> Ignore { get; set; }
		private List<Cluster> Clusters { get; set; }
		private double[,] Distances { get; set; }
		private StreamWriter OutputFile { get; set; }

		class Cluster
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

		public Clustering(string algorithm, int k, Random rand, string ignore)
		{
			Algorithm = algorithm;
			K = k;
			Rand = rand;

			Ignore = new List<int>();
			if (!string.IsNullOrEmpty(ignore))
			{
				var si = ignore.Substring(1, ignore.Length - 2).Split(',');
				foreach (var s in si)
				{
					Ignore.Add(int.Parse(s));
				}
			}
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			Features = new VMatrix(features, 0, 0, features.Rows(), features.Cols());
			if (labels.Data != null)
			{
				Labels = new VMatrix(labels, 0, 0, labels.Rows(), labels.Cols());
			}
			Clusters = new List<Cluster>();
			if (!string.IsNullOrEmpty(OutputFileName))
			{
				OutputFile = File.AppendText(OutputFileName);
				OutputFile.Write("Algorithm: ");
			}

			if (Algorithm == "k")
			{
				if (OutputFile != null)
				{
					OutputFile.WriteLine("k-means (k = " + K + ")");
				}

//				Features.Shuffle(Rand, Labels);

				// create the initial clusters
				for (var k = 0; k < K; k++)
				{
					var cluster = new Cluster(k, Features, k, Ignore);
					Clusters.Add(cluster);
					if (OutputFile != null)
					{
						cluster.PrintCentroid(OutputFile);
					}
				}

				double lastSsd = double.MinValue;

				for (;;)
				{
					var ssd = TrainK();
					if (OutputFile != null)
					{
						OutputFile.WriteLine(string.Format("Sum squared-distance of each row with its centroid={0}", ssd));
					}

					if (ssd != lastSsd)
					{
						lastSsd = ssd;
						if (OutputFile != null)
						{
							OutputFile.WriteLine("Recomputing the centroids of each cluster...");
						}
						foreach (var cluster in Clusters)
						{
							cluster.Recalculate();
							cluster.ClearInstances();
							if (OutputFile != null)
							{
								cluster.PrintCentroid(OutputFile);
							}
						}
					}
					else
					{
						break;
					}
				}
			}
			else if (Algorithm == "single")
			{
				if (OutputFile != null)
				{
					OutputFile.WriteLine("HAC single (k = " + K + ")");
				}

				// create the initial clusters
				for (var row = 0; row < Features.Rows(); row++)
				{
					var cluster = new Cluster(0, Features, row, Ignore);
					cluster.AddInstance(row);
					Clusters.Add(cluster);
				}

				// create the distance matrix
				Distances = new double[Features.Rows(), Features.Rows()];

				for (var row = 0; row < Features.Rows(); row++)
				{
					for (var row2 = row; row2 < Features.Rows(); row2++)
					{
						double distance = 0;
						if (row2 > row)
						{
							distance = Clusters[row].GetDistance(Features.Row(row2));
						}
						Distances[row, row2] = distance;
						if (row != row2)
						{
							Distances[row2, row] = distance;
						}
					}
				}

				int iteration = 0;

				do
				{
					TrainSingle(iteration++);
				} while (Clusters.Count > K);
			}
			else if (Algorithm == "complete")
			{
				if (OutputFile != null)
				{
					OutputFile.WriteLine("HAC complete (k = " + K + ")");
				}

				// create the initial clusters
				for (var row = 0; row < Features.Rows(); row++)
				{
					var cluster = new Cluster(0, Features, row, Ignore);
					cluster.AddInstance(row);
					Clusters.Add(cluster);
				}

				// create the distance matrix
				Distances = new double[Features.Rows(), Features.Rows()];

				for (var row = 0; row < Features.Rows(); row++)
				{
					for (var row2 = row; row2 < Features.Rows(); row2++)
					{
						double distance = 0;
						if (row2 > row)
						{
							distance = Clusters[row].GetDistance(Features.Row(row2));
						}
						Distances[row, row2] = distance;
						if (row != row2)
						{
							Distances[row2, row] = distance;
						}
					}
				}

				int iteration = 0;

				do
				{
					TrainComplete(iteration++);
				} while (Clusters.Count > K);
			}
			else if (Algorithm == "average")
			{
				if (OutputFile != null)
				{
					OutputFile.WriteLine("HAC average (k = " + K + ")");
				}

				// create the initial clusters
				for (var row = 0; row < Features.Rows(); row++)
				{
					var cluster = new Cluster(0, Features, row, Ignore);
					cluster.AddInstance(row);
					Clusters.Add(cluster);
				}

				// create the distance matrix
				Distances = new double[Features.Rows(), Features.Rows()];

				for (var row = 0; row < Features.Rows(); row++)
				{
					for (var row2 = row; row2 < Features.Rows(); row2++)
					{
						double distance = 0;
						if (row2 > row)
						{
							distance = Clusters[row].GetDistance(Features.Row(row2));
						}
						Distances[row, row2] = distance;
						if (row != row2)
						{
							Distances[row2, row] = distance;
						}
					}
				}

				int iteration = 0;

				do
				{
					TrainAverage(iteration++);
				} while (Clusters.Count > K);
			}
			else
			{
				throw new Exception("Inavlid Algorithm - " + Algorithm);
			}

			if (OutputFile != null)
			{
				OutputFile.WriteLine();
				OutputFile.WriteLine("Cluster centroids:");
				
				OutputFile.Write("Cluster#\t\t\t");
				for (var c = 0; c < Clusters.Count; c++)
				{
					OutputFile.Write("\t\t" + c);
				}
				OutputFile.WriteLine();

				OutputFile.Write("# of instances:\t\t\t");
				for (var c = 0; c < Clusters.Count; c++)
				{
					OutputFile.Write("\t\t" + Clusters[c].Instances.Count);
				}
				OutputFile.WriteLine();

				OutputFile.WriteLine("==========================================================================================================");
				for (var col = 0; col < Features.Cols(); col++)
				{
					if (!Ignore.Contains(col))
					{
						OutputFile.Write(Features.AttrName(col));
						foreach (var cluster in Clusters)
						{
							if (cluster.Centroid[col] == Matrix.MISSING)
							{
								OutputFile.Write("\t?");
							}
							else if (Features.ValueCount(col) < 2)
							{
								// continuous
								OutputFile.Write(string.Format("\t{0:0.#####}", cluster.Centroid[col]));
							}
							else
							{
								OutputFile.Write("\t" + Features.AttrValue(col, (int)cluster.Centroid[col]));
							}
						}
						OutputFile.WriteLine();
					}
				}

				double sse = 0;
				OutputFile.Write("Sum squared error:\t");
				foreach (var cluster in Clusters)
				{
					var error = cluster.GetSSE();
					sse += error;
					OutputFile.Write(string.Format("\t{0:0.#####}", error));
				}
				OutputFile.WriteLine();

				OutputFile.WriteLine("Number of clusters: " + Clusters.Count);
				OutputFile.WriteLine(string.Format("Total sum squared error: {0:0.#####}", sse));
				OutputFile.WriteLine(string.Format("DBI: {0}", GetDBI()));
			}

			if (OutputFile != null)
			{
				OutputFile.Close();
			}
		}

		private double TrainK()
		{
			if (OutputFile != null)
			{
				OutputFile.WriteLine("Assigning each row to the cluster of the nearest centroid...");
				OutputFile.WriteLine("The cluster assignments are:");
			}

			// add the training set elements to the clusters
			for (var row = 0; row < Features.Rows(); row++)
			{
				var cluster = GetNearestCluster(Features.Row(row));
				cluster.AddInstance(row);

				if (OutputFile != null)
				{
					if (row % 10 == 0)
					{
						OutputFile.WriteLine();
						OutputFile.Write("\t");
					}
					else
					{
						OutputFile.Write(", ");
					}
					OutputFile.Write(string.Format("{0}={1}", row, cluster.Number));
				}
			}

			if (OutputFile != null)
			{
				OutputFile.WriteLine();
			}

			double sse = 0;

			foreach (var cluster in Clusters)
			{
				sse += cluster.GetSSE();
			}

			return sse;
		}

		private Cluster GetNearestCluster(double[] features)
		{
			Cluster nearest = Clusters[0];
			double nDistance = double.MaxValue;

			foreach (var cluster in Clusters)
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

			if (OutputFile != null)
			{
				OutputFile.WriteLine();
				OutputFile.WriteLine("--------------");
				OutputFile.WriteLine("Iteration " + iteration);
				OutputFile.WriteLine("--------------");
			}

			// find the nearest clusters
			for (var beg = 0; beg < Clusters.Count - 1; beg++)
			{
				for (var end = beg + 1; end < Clusters.Count; end++)
				{
					foreach (var begi in Clusters[beg].Instances)
					{
						foreach (var endi in Clusters[end].Instances)
						{
							if (Distances[begi, endi] < minDist)
							{
								minDist = Distances[begi, endi];
								cluster1 = beg;
								cluster2 = end;
							}
						}
					}
				}
			}

			if (OutputFile != null)
			{
				OutputFile.WriteLine("Merging clusters {0} and {1}\tDistance: {2}", cluster1, cluster2, minDist);
				OutputFile.WriteLine();
			}

			// merge the clusters
			foreach (var i in Clusters[cluster2].Instances)
			{
				Clusters[cluster1].AddInstance(i);
			}

			Clusters[cluster1].Recalculate();
			Clusters.RemoveAt(cluster2);

			for (var c = 0; c < Clusters.Count; c++)
			{
				Clusters[c].Number = c;
				Clusters[c].PrintInstances(OutputFile);
			}

			double sse = 0;

			foreach (var cluster in Clusters)
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

			if (OutputFile != null)
			{
				OutputFile.WriteLine();
				OutputFile.WriteLine("--------------");
				OutputFile.WriteLine("Iteration " + iteration);
				OutputFile.WriteLine("--------------");
			}

			// find the nearest clusters
			for (var beg = 0; beg < Clusters.Count - 1; beg++)
			{
				for (var end = beg + 1; end < Clusters.Count; end++)
				{
					double maxDist = double.MinValue;
					foreach (var begi in Clusters[beg].Instances)
					{
						foreach (var endi in Clusters[end].Instances)
						{
							if (Distances[begi, endi] > maxDist)
							{
								maxDist = Distances[begi, endi];
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

			if (OutputFile != null)
			{
				OutputFile.WriteLine("Merging clusters {0} and {1}\tDistance: {2}", cluster1, cluster2, minDist);
				OutputFile.WriteLine();
			}

			// merge the clusters
			foreach (var i in Clusters[cluster2].Instances)
			{
				Clusters[cluster1].AddInstance(i);
			}

			Clusters[cluster1].Recalculate();
			Clusters.RemoveAt(cluster2);

			for (var c = 0; c < Clusters.Count; c++)
			{
				Clusters[c].Number = c;
				Clusters[c].PrintInstances(OutputFile);
			}

			double sse = 0;

			foreach (var cluster in Clusters)
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

			if (OutputFile != null)
			{
				OutputFile.WriteLine();
				OutputFile.WriteLine("--------------");
				OutputFile.WriteLine("Iteration " + iteration);
				OutputFile.WriteLine("--------------");
			}

			// find the nearest clusters
			for (var beg = 0; beg < Clusters.Count - 1; beg++)
			{
				for (var end = beg + 1; end < Clusters.Count; end++)
				{
					double totDist = 0;
					int count = 0;
					foreach (var begi in Clusters[beg].Instances)
					{
						foreach (var endi in Clusters[end].Instances)
						{
							totDist += Distances[begi, endi];
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

			if (OutputFile != null)
			{
				OutputFile.WriteLine("Merging clusters {0} and {1}\tDistance: {2}", cluster1, cluster2, minDist);
				OutputFile.WriteLine();
			}

			// merge the clusters
			foreach (var i in Clusters[cluster2].Instances)
			{
				Clusters[cluster1].AddInstance(i);
			}

			Clusters[cluster1].Recalculate();
			Clusters.RemoveAt(cluster2);

			for (var c = 0; c < Clusters.Count; c++)
			{
				Clusters[c].Number = c;
				Clusters[c].PrintInstances(OutputFile);
			}

			double sse = 0;

			foreach (var cluster in Clusters)
			{
				sse += cluster.GetSSE();
			}

			return sse;
		}

		private double GetDBI()
		{
			double dbi = 0;

			for (var c1 = 0; c1 < Clusters.Count - 1; c1++)
			{
				double S1 = Clusters[c1].GetSSE() / Clusters[c1].Instances.Count;
				double maxR = double.MinValue;
				for (var c2 = c1 + 1; c2 < Clusters.Count; c2++)
				{
					double S2 = Clusters[c2].GetSSE() / Clusters[c2].Instances.Count;
					double d = Clusters[c1].GetError(Clusters[c2].Centroid);
					double R = (S1 + S2) / d;
					if (R > maxR)
					{
						maxR = R;
					}
				}

				dbi += maxR;
			}

			return dbi / Clusters.Count;
		}

		public override void Predict(double[] features, double[] labels)
		{
			var nearest = GetNearestCluster(features);
			GetOutput(nearest, labels);
		}

		private void GetOutput(Cluster cluster, double[] labels)
		{
			bool isContinuous = Labels.ValueCount(0) < 2;
			int[] count = new int[isContinuous ? 1 : Labels.ValueCount(0)];
			double result = 0;

			// calculate the output
			for (var i = 0; i < cluster.Instances.Count; i++)
			{
				if (isContinuous)
				{
					// continuous
					result += Labels.Get(cluster.Instances[i], 0);
					count[0]++;
				}
				else
				{
					double idx = Labels.Get(cluster.Instances[i], 0);
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
