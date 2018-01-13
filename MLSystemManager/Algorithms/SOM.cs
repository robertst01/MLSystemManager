#define parallel
#define toroid

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	/// <summary>
	/// Self Organizing Map (Kohonen Network)
	/// </summary>
	public class SOM : SupervisedLearner
	{
		private List<List<Node>> m_layers;
		private Random m_rand;
		private double m_rate = 0.1;
		private int m_iterations = 0;						// number of iterations to train
		private int m_gridSize = 0;							// dimensions of the grid
		private StreamWriter m_outputFile = null;

		private class Node
		{
			public int row { get; set; }					// the row for this node
			public int col { get; set; }					// the column for this node
			public double distance { get; set; }			// the distance to the input vector
			public double[] weights { get; set; }			// the weights for all input nodes
			public int[] outputs { get; set; }				// the outputs for this node
			public int output
			{
				get
				{
					if ((outputs == null) || (outputs.Length < 1))
					{
						return -1;
					}

					int output = 0;
					int maxOut = outputs[0];

					for (var v = 1; v < outputs.Length; v++)
					{
						if (outputs[v] > maxOut)
						{
							output = v;
							maxOut = outputs[v];
						}
					}

					if (maxOut == 0)
					{
						output = -1;
					}

					return output;
				}

				set
				{
					if ((outputs != null) && (outputs.Length > value))
					{
						outputs[value]++;
					}
				}
			}

			public Node(int row, int col, int numWeights, int numOutputs, Random rand)
			{
				this.row = row;
				this.col = col;

				weights = new double[numWeights];
				for (var i = 0; i < numWeights; i++)
				{
					weights[i] = rand.NextDouble();
				}

				outputs = new int[numOutputs];
				for (var i = 0; i < numOutputs; i++)
				{
					outputs[i] = 0;
				}
			}
		}

		public SOM()
		{
			m_rand = new Random();
			m_layers = new List<List<Node>>();
		}

		public SOM(Parameters parameters)
		{
			m_rand = Rand.Get();
			m_rate = parameters.Rate;
			m_iterations = parameters.Iterations;
			m_gridSize = parameters.GridSize;
			m_layers = new List<List<Node>>();
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			int numWeights = features.Cols();

			// add the nodes
			List<Node> nodes = new List<Node>();

			for (int row = 0; row < m_gridSize; row++)
			{
				for (int col = 0; col < m_gridSize; col++)
				{
					var labelValueCount = labels.ValueCount(0);

					if (labelValueCount < 2)
					{
						// continuous
						throw new Exception("Output must be nominal");
					}
					else
					{
						nodes.Add(new Node(row, col, numWeights, labelValueCount, m_rand));
					}
				}
			}

			m_layers.Add(nodes);

			if (!string.IsNullOrEmpty(OutputFileName))
			{
				m_outputFile = File.AppendText(OutputFileName);
			}

			if (m_outputFile != null)
			{
				m_outputFile.WriteLine(string.Format("Grid size: {0}", m_gridSize));
				m_outputFile.WriteLine(string.Format("Iterations: {0}", m_iterations));
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
			}
			
			TrainEpoch(features, labels);

			//Console.WriteLine();
			//for (int row = 0; row < m_gridSize; row++)
			//{
			//	for (int col = 0; col < m_gridSize; col++)
			//	{
			//		int n = (row * m_gridSize) + col;
			//		int output = (int)m_layers[0][n].output;
			//		if (output >= 0)
			//		{
			//			Console.Write(output);
			//		}
			//		else
			//		{
			//			Console.Write('_');
			//		}
			//	}
			//	Console.WriteLine();
			//}

			FixOutputs();

			Console.WriteLine();
			for (int row = 0; row < m_gridSize; row++)
			{
				for (int col = 0; col < m_gridSize; col++)
				{
					int n = (row * m_gridSize) + col;
					int output = (int)m_layers[0][n].output;
					if (output >= 0)
					{
						Console.Write(output);
					}
					else
					{
						Console.Write('_');
					}
				}
				Console.WriteLine();
			}

			if (m_outputFile != null)
			{
				m_outputFile.WriteLine();
				for (int row = 0; row < m_gridSize; row++)
				{
					for (int col = 0; col < m_gridSize; col++)
					{
						int n = (row * m_gridSize) + col;
						int output = (int)m_layers[0][n].output;
						if (output >= 0)
						{
							m_outputFile.Write(output);
						}
						else
						{
							m_outputFile.Write('_');
						}
					}
					m_outputFile.WriteLine();
				}

				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
			}

			if (m_outputFile != null)
			{
				m_outputFile.Close();
			}
		}

		private void TrainEpoch(VMatrix features, VMatrix labels)
		{
			double minDistance;
			Node bmu;
			object lo = new object();

			Console.Write("TrainEpoch ");
			int cl = Console.CursorLeft;

			if (m_iterations < 1)
			{
				m_iterations = features.Rows() * 10;
			}

			double mapRadius = (double)m_gridSize / 2;
			double timeConstant = (double)m_iterations / Math.Log(mapRadius);

			for (int iteration = 0; iteration < m_iterations; iteration++)
			{
				int row = m_rand.Next(features.Rows());
				minDistance = double.MaxValue;
				bmu = null;

				if (((iteration % 100) == 0) || (iteration == (m_iterations - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(iteration);
				}

				// calculate the distance
#if parallel
				Parallel.ForEach(m_layers[0], node =>
#else
				foreach(var node in m_layers[0])
#endif
				{
					node.distance = 0;

					// calculate the distance
					for (var w = 0; w < node.weights.Length; w++)
					{
						node.distance += (features.Get(row, w) - node.weights[w]) * (features.Get(row, w) - node.weights[w]);
					}

					lock (lo)
					{
						if (node.distance < minDistance)
						{
							minDistance = node.distance;
							bmu = node;
						}
					}
#if parallel
				});
#else
				}
#endif

				bmu.output = (int)labels.Get(row, 0);

				// calculate the error and weight changes
				double neighborhoodRadius = mapRadius * Math.Exp(-(double)iteration / timeConstant);
				double radiusSquared = neighborhoodRadius * neighborhoodRadius;
				double learningRate = m_rate * Math.Exp(-(double)iteration / (m_iterations - iteration));

#if parallel
				Parallel.ForEach(m_layers[0], node =>
#else
				foreach (var node in m_layers[0])
#endif
				{
					int dX = Math.Abs(node.col - bmu.col);
#if toroid
					if (dX > (m_gridSize / 2))
					{
						dX = m_gridSize - dX;
					}
#endif
					int dY = Math.Abs(node.row - bmu.row);
#if toroid
					if (dY > (m_gridSize / 2))
					{
						dY = m_gridSize - dY;
					}
#endif
					double distSquared = ((dX * dX) + (dY * dY));
					if (distSquared <= neighborhoodRadius /*radiusSquared*/)
					{
						double influence = Math.Exp(-distSquared / (2 * radiusSquared));
						for (int w = 0; w < node.weights.Length; w++)
						{
							node.weights[w] += influence * learningRate * (features.Get(row, w) - node.weights[w]);
						}
					}
#if parallel
				});
#else
				}
#endif
			}

			Console.WriteLine();
		}

		private void FixOutputs()
		{
			// TODO - need to change this to use circular neighborhood and don't update all until the end
#if parallel
			Parallel.ForEach(m_layers[0], node =>
#else
			foreach (var node in m_layers[0])
#endif
			{
				int offset = 1;

				// fix the output if it's less than zero
				while (node.output < 0)
				{
					int begRow = node.row - offset;
					if (begRow < 0)
					{
						begRow = 0;
					}

					int endRow = node.row + offset;
					if (endRow >= m_gridSize)
					{
						endRow = m_gridSize - 1;
					}

					int begCol = node.col - offset;
					if (begCol < 0)
					{
						begCol = 0;
					}

					int endCol = node.col + offset;
					if (endCol >= m_gridSize)
					{
						endCol = m_gridSize - 1;
					}

					// add up all the outputs in this area
					for (int row = begRow; row <= endRow; row++)
					{
						for (int col = begCol; col <= endCol; col++)
						{
							int n = (row * m_gridSize) + col;
							int output = m_layers[0][n].output;
							if (output >= 0)
							{
								node.output = output;
							}
						}
					}

					offset++;
				}
#if parallel
			});
#else
			}
#endif
		}

		// Calculate the MSE
		public override double GetMSE(Matrix features, Matrix labels)
		{
			return 0;
		}

		// Calculate the MSE
		public override double VGetMSE(VMatrix features, VMatrix labels)
		{
			double sse = 0;
			double minDistance;
			Node bmu;
			object lo = new object();

			Console.Write("VGetMSE ");
			int cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				if (((row % 10) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

				minDistance = double.MaxValue;
				bmu = null;

				// calculate the distance
#if parallel
				Parallel.ForEach(m_layers[0], node =>
#else
				foreach (var node in m_layers[0])
#endif
				{
					node.distance = 0;

					// calculate the distance
					for (var w = 0; w < node.weights.Length; w++)
					{
						node.distance += (features.Get(row, w) - node.weights[w]) * (features.Get(row, w) - node.weights[w]);
					}

					lock (lo)
					{
						if (node.distance < minDistance)
						{
							minDistance = node.distance;
							bmu = node;
						}
					}
#if parallel
				});
#else
				}
#endif

				// calculate the error of the output layer
				if (bmu.output >= 0)
				{
					double target = labels.Get(row, 0);
					var error = target - bmu.output;

					// update the error
					sse += error * error;
				}
			}

			Console.WriteLine();

			return sse / features.Rows();
		}

		private void PrintWeights()
		{
			for (var layer = 0; layer < m_layers.Count; layer++)
			{
				m_outputFile.WriteLine("Layer " + layer);
				foreach (var node in m_layers[layer])
				{
					for (var w = 0; w < node.weights.Length - 1; w++)
					{
						m_outputFile.Write(string.Format("{0}\t", node.weights[w]));
					}
					m_outputFile.WriteLine(node.weights[node.weights.Length - 1]);
				}
			}
			m_outputFile.WriteLine();
		}

		public override void Predict(double[] features, double[] labels)
		{
			double minDistance = double.MaxValue;
			Node bmu = null;
			object lo = new object();

			// calculate the distance
#if parallel
			Parallel.ForEach(m_layers[0], node =>
#else
			foreach (var node in m_layers[0])
#endif
			{
				node.distance = 0;

				// calculate the distance
				for (var w = 0; w < node.weights.Length; w++)
				{
					node.distance += (features[w] - node.weights[w]) * (features[w] - node.weights[w]);
				}

				lock (lo)
				{
					if (node.distance < minDistance)
					{
						minDistance = node.distance;
						bmu = node;
					}
				}
#if parallel
				});
#else
			}
#endif

			labels[0] = bmu.output;
		}
	}
}
