#define parallel

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	public class ReLU : SupervisedLearner
	{
		private List<List<Node>> m_layers;
		private Random m_rand;
		private double m_rate = 0.1;
		private double m_momentum = 0.9;
		private int[] m_hidden = null;
		private List<double[]> m_weights = null;
		private List<InputFeature> m_inputFeatures = null;
		private List<OutputLabel> m_outputLabels = null;
		private StreamWriter m_outputFile = null;

		class Node
		{
			public int index { get; set; }					// the index for this node
			public double net { get; set; }					// the net for this node
			public double output { get; set; }				// the output of this node
			public double error { get; set; }				// the error for this node
			public double[] weights { get; set; }			// the weights for all nodes connected to this node
			public double[] bestWeights { get; set; }		// the best weights so far
			public double[] deltas { get; set; }			// weight deltas from previous epoch
			public bool isContinuous { get; set; }			// true if the outout is continuous
			public int labelCol { get; set; }				// the label column that this output node corresponds to
			public double labelVal { get; set; }			// the value of the label column that this output node corresponds to

			public Node(int numWeights, bool isContinuous, int labelCol, double labelVal, Random rand, double[] w)
			{
				index = 0;

				weights = new double[numWeights];
				bestWeights = new double[numWeights];
				for (var i = 0; i < numWeights; i++)
				{
					if (w != null)
					{
						weights[i] = w[i];
					}
					else
					{
						weights[i] = 0.1 - (rand.NextDouble() * 0.2);
					}
					bestWeights[i] = weights[i];
				}

				deltas = new double[numWeights];
				for (var i = 0; i < numWeights; i++)
				{
					deltas[i] = 0;
				}

				this.isContinuous = isContinuous;
				this.labelCol = labelCol;
				this.labelVal = labelVal;
			}

			public void SaveBestWeights()
			{
				for (var i = 0; i < weights.Length; i++)
				{
					bestWeights[i] = weights[i];
				}
			}

			public void RestoreBestWeights()
			{
				for (var i = 0; i < weights.Length; i++)
				{
					weights[i] = bestWeights[i];
				}
			}
		}

		class InputFeature
		{
			public int feature;
			public int valueCount;
			public double minValue;
			public double maxValue;

			public InputFeature(int feature, int valueCount, double minValue, double maxValue)
			{
				this.feature = feature;
				this.valueCount = valueCount;
				this.minValue = minValue;
				this.maxValue = maxValue;
			}
		}

		class OutputLabel
		{
			public int label;
			public int valueCount;
			public double value;

			public OutputLabel(int label, int valueCount, double value)
			{
				this.label = label;
				this.valueCount = valueCount;
				this.value = value;
			}
		}

		public ReLU()
		{
			m_rand = new Random();
			m_layers = new List<List<Node>>();
		}

		public ReLU(Parameters parameters)
		{
			m_rand = Rand.Get();
			m_rate = parameters.Rate;
			m_momentum = parameters.Momentum;
			m_hidden = parameters.Hidden;
			m_layers = new List<List<Node>>();
		}

		public bool LoadWeights()
		{
			List<double[]> weights = null;

			if (!string.IsNullOrEmpty(WeightsFileName) && File.Exists(WeightsFileName))
			{
				weights = new List<double[]>();
				int prevNodeCount = 0;
				string[] nodeCounts = null;

				using (StreamReader file = new StreamReader(WeightsFileName))
				{
					for (; ; )
					{
						// read the nodes info
						String line = file.ReadLine();
						if (!string.IsNullOrEmpty(line) && !line.StartsWith("%"))
						{
							// parse the nodes array
							nodeCounts = line.Split(',');

							// fix the hidden node counts
							m_hidden = new int[nodeCounts.Length - 2];

							break;
						}
					}

					int inputCount = int.Parse(nodeCounts[0].Trim());
					m_inputFeatures = new List<InputFeature>();
					for (; ; )
					{
						// read the input node params
						String line = file.ReadLine();
						if (!string.IsNullOrEmpty(line) && !line.StartsWith("%"))
						{
							var inputs = line.Split(',');
							var feature = new InputFeature(int.Parse(inputs[0].Trim()), int.Parse(inputs[1].Trim()), double.Parse(inputs[2].Trim()), double.Parse(inputs[3].Trim()));
							m_inputFeatures.Add(feature);

							if (m_inputFeatures.Count >= inputCount)
							{
								break;
							}
						}
					}

					int outputCount = int.Parse(nodeCounts[nodeCounts.Length - 1].Trim());
					m_outputLabels = new List<OutputLabel>();
					for (; ; )
					{
						// read the output node params
						String line = file.ReadLine();
						if (!string.IsNullOrEmpty(line) && !line.StartsWith("%"))
						{
							var outputs = line.Split(',');
							var label = new OutputLabel(int.Parse(outputs[0].Trim()), int.Parse(outputs[1].Trim()), double.Parse(outputs[2].Trim()));
							m_outputLabels.Add(label);

							if (m_outputLabels.Count >= outputCount)
							{
								break;
							}
						}
					}

					// read the weights
					for (int layer = 0; layer < nodeCounts.Length; layer++)
					{
						if (layer == 0)
						{
							// input layer
							prevNodeCount = int.Parse(nodeCounts[layer].Trim()) + 1;
						}
						else
						{
							int nodes = int.Parse(nodeCounts[layer].Trim());
							for (int n = 0; n < nodes; n++)
							{
								double[] w = new double[prevNodeCount];
								var line = file.ReadLine();
								if (!string.IsNullOrEmpty(line) && !line.StartsWith("%"))
								{
									string[] ws = line.Split(',');
									if (ws.Length != prevNodeCount)
									{
										Console.WriteLine(string.Format("Incorrect weight count (layer {0}, node {1}, count {2}", layer, n, ws.Length));
										Environment.Exit(0);
									}
									for (int i = 0; i < ws.Length; i++)
									{
										w[i] = double.Parse(ws[i]);
									}

									weights.Add(w);
								}
								else
								{
									n--;
								}
							}

							if (layer < nodeCounts.Length - 1)
							{
								// hidden layer
								m_hidden[layer - 1] = nodes;
							}

							prevNodeCount = nodes + 1;
						}
					}
				}

				if (weights.Count > 0)
				{
					m_weights = weights;

					m_layers = new List<List<Node>>();
					int prevNodes = int.Parse(nodeCounts[0].Trim()) + 1;
					int wIdx = 0;							// index into the weights array

					for (var layer = 0; layer <= m_hidden.Length; layer++)
					{
						// add the nodes for this layer
						List<Node> nodes = new List<Node>();

						if (layer < m_hidden.Length)
						{
							// hidden layer
							for (var n = 0; n < m_hidden[layer]; n++)
							{
								nodes.Add(new Node(prevNodes, false, 0, 0, m_rand, m_weights[wIdx++]));
							}
						}
						else
						{
							// output layer
							for (var n = 0; n < m_outputLabels.Count; n++)
							{
								var labelValueCount = m_outputLabels[n].valueCount;

								if (labelValueCount < 2)
								{
									// continuous
									nodes.Add(new Node(prevNodes, true, n, -1, m_rand, m_weights[wIdx++]));
								}
								else
								{
									nodes.Add(new Node(prevNodes, false, m_outputLabels[n].label, m_outputLabels[n].value, m_rand, m_weights[wIdx++]));
								}
							}
						}

						prevNodes = nodes.Count + 1;

						m_layers.Add(nodes);
					}
				}
			}

			return (weights != null) && (weights.Count > 0);
		}

		private void InitNodes()
		{
			for (var layer = 0; layer < m_layers.Count - 1; layer++)
			{
				for (int idx = 0; idx < m_layers[layer].Count; idx++)
				{
					m_layers[layer][idx].index = idx;
				}
			}
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			if (m_hidden.Length < 1)
			{
				m_hidden = new int[1] { features.Cols() * 2 };
			}

			var weightsLoaded = LoadWeights();
			if (!weightsLoaded)
			{
				int prevNodes = features.Cols() + 1;
				int wIdx = 0;							// index into the weights array

				for (var layer = 0; layer <= m_hidden.Length; layer++)
				{
					// add the nodes for this layer
					List<Node> nodes = new List<Node>();

					if (layer < m_hidden.Length)
					{
						// hidden layer
						for (var n = 0; n < m_hidden[layer]; n++)
						{
							if (m_weights != null)
							{
								nodes.Add(new Node(prevNodes, false, 0, 0, m_rand, m_weights[wIdx++]));
							}
							else
							{
								nodes.Add(new Node(prevNodes, false, 0, 0, m_rand, null));
							}
						}
					}
					else
					{
						// output layer - figure out how many outputs we need
						for (var col = 0; col < labels.Cols(); col++)
						{
							var labelValueCount = labels.ValueCount(col);

							if (labelValueCount < 2)
							{
								// continuous
								if (m_weights != null)
								{
									nodes.Add(new Node(prevNodes, true, col, -1, m_rand, m_weights[wIdx++]));
								}
								else
								{
									nodes.Add(new Node(prevNodes, true, col, -1, m_rand, null));
								}
							}
							else
							{
								for (var n = 0; n < labelValueCount; n++)
								{
									if (m_weights != null)
									{
										nodes.Add(new Node(prevNodes, false, col, n, m_rand, m_weights[wIdx++]));
									}
									else
									{
										nodes.Add(new Node(prevNodes, false, col, n, m_rand, null));
									}
								}
							}
						}
					}

					prevNodes = nodes.Count + 1;

					m_layers.Add(nodes);
				}
			}

			InitNodes();

			if (!string.IsNullOrEmpty(OutputFileName))
			{
				m_outputFile = File.AppendText(OutputFileName);
			}

			int trainSize = (int)(0.75 * features.Rows());
			VMatrix trainFeatures = new VMatrix(features, 0, 0, trainSize, features.Cols());
			VMatrix trainLabels = new VMatrix(labels, 0, 0, trainSize, labels.Cols());
			VMatrix validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			VMatrix validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			int epoch = 0;							// current epoch number
			double bestTrainMSE = double.MaxValue;	// best training MSE so far
			double bestMSE = double.MaxValue;		// best validation MSE so far
			double bestAccuracy = double.MaxValue;	// best validationa accuracy so far
			double initialMSE = double.MaxValue;	// MSE for first epoch
			int eCount = 0;							// number of epochs since the best MSE
			int bestEpoch = 0;						// epoch number of best MSE
			bool done = false;
			bool checkDone = false;					// if true, check to see if we're done

			Console.WriteLine("Epoch\tMSE (training)\t\tMSE (validation)\taccuracy (validation)");
			if (m_outputFile != null)
			{
				m_outputFile.WriteLine(string.Format("{0} layers, {1} output nodes", m_layers.Count, m_layers[m_layers.Count - 1].Count));
				m_outputFile.WriteLine("Momentum: " + m_momentum);
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
				m_outputFile.WriteLine("Epoch\tMSE (training)\t\tMSE (validation)\taccuracy (validation)");
			}

			do
			{
				// shuffle the training set
				trainFeatures.Shuffle(m_rand, trainLabels);

				double trainMSE;

				if (m_weights != null)
				{
					// not training
					trainMSE = VGetMSE(trainFeatures, trainLabels);
					epoch++;
				}
				else
				{
					trainMSE = TrainEpoch(++epoch, trainFeatures, trainLabels);
				}

				// check the MSE after this epoch
				double mse = VGetMSE(validationFeatures, validationLabels);

				// check the validation accuracy after this epoch
				double accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);

				Console.WriteLine(string.Format("{0}\t{1}\t{2}\t{3}", epoch, trainMSE, mse, accuracy));
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine(string.Format("{0}\t{1}\t{2}\t{3}", epoch, trainMSE, mse, accuracy));
				}

				if (m_weights != null)
				{
					// not really training
					done = true;
				}
				else if (mse == 0.0)
				{
					// can't get better than this
					done = true;
				}
				else if ((epoch == 1) || (mse <= bestMSE))
				{
					if (epoch == 1)
					{
						// save the initial MSE
						initialMSE = mse;
					}
					else if (!checkDone && (mse < initialMSE * 0.9))
					{
						checkDone = true;
					}

					// save the best for later
					bestTrainMSE = trainMSE;
					bestMSE = mse;
					bestAccuracy = accuracy;
					bestEpoch = epoch;
					eCount = 0;
					for (var layer = 0; layer < m_layers.Count - 1; layer++)
					{
						foreach (var node in m_layers[layer])
						{
							node.SaveBestWeights();
						}
					}
				}
				else if (checkDone)
				{
					// check to see if we're done
					eCount++;
					if (eCount >= 20)
					{
						done = true;
					}
				}
				else if (mse > initialMSE * 1.1)
				{
					// are we getting really worse?
					checkDone = true;
				}
				else if (epoch >= 10000)
				{
					// time to stop
					done = true;
				}
			} while (!done);

			if (m_outputFile != null)
			{
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
			}

			if ((bestEpoch > 0) && (bestEpoch != epoch))
			{
				for (var layer = 0; layer < m_layers.Count - 1; layer++)
				{
					foreach (var node in m_layers[layer])
					{
						node.RestoreBestWeights();
					}
				}
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine();
					m_outputFile.WriteLine(string.Format("Best Weights (from Epoch {0}, trainMSE={1}, valMSE={2}, valAcc={3})", bestEpoch, bestTrainMSE, bestMSE, bestAccuracy));
					PrintWeights();
				}
			}

			if (m_outputFile != null)
			{
				m_outputFile.Close();
			}

			// save the weights
			if (!weightsLoaded && !string.IsNullOrEmpty(WeightsFileName))
			{
				using (StreamWriter wf = new StreamWriter(WeightsFileName))
				{
					// write the node count
					wf.Write("% #input nodes, ");
					for (var layer = 0; layer < m_layers.Count - 1; layer++)
					{
						wf.Write(string.Format("#hidden{0} nodes, ", layer + 1));
					}
					wf.WriteLine("#output nodes");
					for (var layer = 0; layer < m_layers.Count; layer++)
					{
						if (layer == 0)
						{
							// write the input node count
							wf.Write(m_layers[0][0].weights.Length - 1);
						}
						wf.Write(string.Format(",{0}", m_layers[layer].Count));
					}
					wf.WriteLine();
					wf.WriteLine();

					// write the input params
					wf.WriteLine("% feature, valueCount, min value, max value");
					for (var col = 0; col < trainFeatures.Cols(); col++)
					{
						wf.WriteLine(string.Format("{0},{1},{2},{3}", col, trainFeatures.ValueCount(col), colMin[col], colMax[col]));
					}
					wf.WriteLine();

					// write the output params
					wf.WriteLine("% label, valueCount, value");
					foreach (var node in m_layers[m_layers.Count - 1])
					{
						wf.WriteLine(string.Format("{0},{1},{2}", node.labelCol, labels.ValueCount(node.labelCol), node.labelVal));
					}

					// write the weights
					for (var layer = 0; layer < m_layers.Count; layer++)
					{
						wf.WriteLine();
						if (layer < m_layers.Count - 1)
						{
							wf.WriteLine(string.Format("% hidden{0} weights", layer + 1));
						}
						else
						{
							wf.WriteLine("% output weights");
						}
						foreach (var node in m_layers[layer])
						{
							for (var w = 0; w < node.weights.Length - 1; w++)
							{
								wf.Write(string.Format("{0},", node.weights[w]));
							}
							wf.WriteLine(node.weights[node.weights.Length - 1]);
						}
					}
				}
			}
		}

		private double TrainEpoch(int epoch, VMatrix features, VMatrix labels)
		{
			double sse = 0;
			object lo = new object();

			Console.Write("TrainEpoch ");
			int cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				if (((row % 100) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

				// calculate the output
				for (var layer = 0; layer < m_layers.Count; layer++)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						node.net = 0;

						// calculate the net value
						for (var w = 0; w < node.weights.Length - 1; w++)
						{
							if (layer == 0)
							{
								node.net += node.weights[w] * features.Get(row, w);
							}
							else
							{
								node.net += node.weights[w] * m_layers[layer - 1][w].output;
							}
						}
						// add the bias
						node.net += node.weights[node.weights.Length - 1];

						node.output = Activation(node.net);
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error and weight changes
				for (var layer = m_layers.Count - 1; layer >= 0; layer--)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						double fPrime = FPrime(node.output);
						if (layer == m_layers.Count - 1)
						{
							// output layer
							double target = labels.Get(row, node.labelCol);
							if (!node.isContinuous)
							{
								// nominal
								if (target == node.labelVal)
								{
									target = 0.9;
								}
								else
								{
									target = 0.1;
								}
							}

							var error = target - node.output;
							node.error = error * fPrime;
							lock (lo) { sse += error * error; }
						}
						else
						{
							// hidden layer
							double sum = 0;
							foreach (var tn in m_layers[layer + 1])
							{
								sum += tn.error * tn.weights[node.index];
							}
							node.error = sum * fPrime;
						}

						// calculate the weight changes
						double delta;
						for (var w = 0; w < node.weights.Length - 1; w++)
						{
							if (layer == 0)
							{
								delta = m_rate * node.error * features.Get(row, w);
							}
							else
							{
								delta = m_rate * node.error * m_layers[layer - 1][w].output;
							}
							delta += m_momentum * node.deltas[w];
							node.deltas[w] = delta;
						}

						// calculate the bias weight change
						delta = m_rate * node.error;
						delta += m_momentum * node.deltas[node.weights.Length - 1];
						node.deltas[node.weights.Length - 1] = delta;
#if parallel
					});
#else
					}
#endif
				}

				// update the weights
				for (var layer = 0; layer < m_layers.Count; layer++)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						for (var weight = 0; weight < node.weights.Length; weight++)
						{
							node.weights[weight] += node.deltas[weight];
						}
#if parallel
					});
#else
					}
#endif
				}
			}

			Console.WriteLine();

			return sse / features.Rows();
		}

		private double Activation(double net)
		{
			return (net < 0 ? 0.01 * net : net);
		}

		private double FPrime(double net)
		{
			return (net < 0 ? 0.01 : 1.0);
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

			Console.Write("VGetMSE ");
			int cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				if (((row % 10) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

				// calculate the output
				for (var layer = 0; layer < m_layers.Count; layer++)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						node.net = 0;

						// calculate the net value
						for (var w = 0; w < node.weights.Length - 1; w++)
						{
							if (layer == 0)
							{
								node.net += node.weights[w] * features.Get(row, w);
							}
							else
							{
								node.net += node.weights[w] * m_layers[layer - 1][w].output;
							}
						}
						// add the bias
						node.net += node.weights[node.weights.Length - 1];

						node.output = Activation(node.net);
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error of the output layer
				for (var n = 0; n < m_layers[m_layers.Count - 1].Count; n++)
				{
					var node = m_layers[m_layers.Count - 1][n];
					double target = labels.Get(row, node.labelCol);
					if (!node.isContinuous)
					{
						// nominal
						if (target == node.labelVal)
						{
							target = 0.9;
						}
						else
						{
							target = 0.1;
						}
					}
					var error = target - node.output;

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
			for (var layer = 0; layer < m_layers.Count; layer++)
			{
#if parallel
				Parallel.ForEach(m_layers[layer], node =>
#else
				foreach (var node in m_layers[layer])
#endif
				{
					node.net = 0;

					// calculate the net value
					for (var w = 0; w < node.weights.Length - 1; w++)
					{
						if (layer == 0)
						{
							node.net += node.weights[w] * features[w];
						}
						else
						{
							node.net += node.weights[w] * m_layers[layer - 1][w].output;
						}
					}
					// add the bias
					node.net += node.weights[node.weights.Length - 1];

					node.output = Activation(node.net);
#if parallel
				});
#else
				}
#endif
			}

			int labelIdx = 0;
			for (var n = 0; n < m_layers[m_layers.Count - 1].Count; n++)
			{
				var node = m_layers[m_layers.Count - 1][n];

				if (node.isContinuous)
				{
					labels[labelIdx++] = node.output;
				}
				else
				{
					// find the max output for this labelCol
					double max = node.output;
					var labelCol = node.labelCol;
					double labelVal = node.labelVal;
					int nIdx;
					for (nIdx = 1; nIdx + n < m_layers[m_layers.Count - 1].Count; nIdx++)
					{
						var tn = m_layers[m_layers.Count - 1][n + nIdx];
						if (tn.labelCol != labelCol)
						{
							break;
						}
						else if (tn.output > max)
						{
							max = tn.output;
							labelVal = tn.labelVal;
						}
					}
					labels[labelIdx++] = labelVal;

					// skip to the next label
					n += nIdx - 1;
				}
			}
		}
	}
}
