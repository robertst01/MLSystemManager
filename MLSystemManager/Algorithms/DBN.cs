//#define parallel

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	/// <summary>
	/// Deep Belief Network
	/// </summary>
	class DBN : SupervisedLearner
	{
		private List<List<Node>> m_layers;
		private Random m_rand;
		private double m_rate = 0.1;
		private double m_momentum = 0.9;
		private double m_pi = 0.8;
		private double m_ph = 0.5;
		private bool m_sample = false;
		private int[] m_hidden = null;
		private List<double[]> m_weights = null;
		private List<OutputLabel> m_outputLabels = null;
		private StreamWriter m_outputFile = null;

		public class Node
		{
			public int index { get; set; }					// the index of this node
			public double net { get; set; }					// the net for this node
			public double output { get; set; }				// the output of this node
			public double sample { get; set; }				// the sample of this node (0 or 1)
			public double net2 { get; set; }				// the net for this node (dbm training)
			public double output2 { get; set; }				// the output of this node (dbm training)
			public double sample2 { get; set; }				// the sample of this node (0 or 1) (dbm training)
			public double error { get; set; }				// the error for this node
			public double[] weights { get; set; }			// the weights for all nodes connected to this node
			public double[] bestWeights { get; set; }		// the best weights so far
			public double[] deltas { get; set; }			// weight deltas from previous epoch
			public bool isActive { get; set; }				// true if this node is currently active

			public Node()
			{
				index = 0;
				net = 0;
				output = 0;
				error = 0;
				weights = null;
				bestWeights = null;
				deltas = null;
				isActive = true;
			}

			public Node(int numWeights, Random rand, double[] w)
			{
				if (numWeights > 0)
				{
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
				}

				this.isActive = true;
			}

			public void SaveBestWeights()
			{
				if (weights != null)
				{
					for (var i = 0; i < weights.Length; i++)
					{
						bestWeights[i] = weights[i];
					}
				}
			}

			public void RestoreBestWeights()
			{
				if (weights != null)
				{
					for (var i = 0; i < weights.Length; i++)
					{
						weights[i] = bestWeights[i];
					}
				}
			}
		}

		public class InputNode : Node
		{
			public int feature { get; set; }
			public int valueCount { get; set; }
			public double minValue { get; set; }
			public double maxValue { get; set; }
			public InputNode(int feature, int valueCount, double minValue, double maxValue, Random rand)
				: base(1, rand, null)
			{
				this.feature = feature;
				this.valueCount = valueCount;
				this.minValue = minValue;
				this.maxValue = maxValue;
			}
		}

		public class HiddenNode : Node
		{
			public HiddenNode(int numWeights, Random rand, double[] w)
				: base(numWeights, rand, w)
			{
			}
		}

		public class OutputNode : Node
		{
			public bool isContinuous { get; set; }			// true if the outout is continuous
			public int labelCol { get; set; }				// the label column that this output node corresponds to
			public double labelVal { get; set; }			// the value of the label column that this output node corresponds to
			public OutputNode(int numWeights, bool isContinuous, int labelCol, double labelVal, Random rand, double[] w)
				: base(numWeights, rand, w)
			{
				this.isContinuous = isContinuous;
				this.labelCol = labelCol;
				this.labelVal = labelVal;
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

		public DBN()
		{
			m_rand = new Random();
			m_layers = new List<List<Node>>();
		}

		public DBN(Random rand, double rate, double momentum, double ph, double pi, bool sample, int[] hidden)
		{
			m_rand = rand;
			m_rate = rate;
			m_momentum = momentum;
			m_ph = ph;
			m_pi = pi;
			m_sample = sample;
			m_hidden = hidden;
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
				List<Node> iNodes = new List<Node>();

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
					for (; ; )
					{
						// read the input node params
						String line = file.ReadLine();
						if (!string.IsNullOrEmpty(line) && !line.StartsWith("%"))
						{
							var inputs = line.Split(',');
							var node = new InputNode(int.Parse(inputs[0].Trim()), int.Parse(inputs[1].Trim()), double.Parse(inputs[2].Trim()), double.Parse(inputs[3].Trim()), m_rand);
							iNodes.Add(node);

							if (iNodes.Count >= inputCount)
							{
								break;
							}
						}
					}

					int outputCount = int.Parse(nodeCounts[nodeCounts.Length - 1].Trim());
					m_outputLabels = new List<OutputLabel>();
					for (;;)
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

					// add the input nodes
					m_layers.Add(iNodes);

					// add the hidden nodes
					for (var layer = 0; layer < m_hidden.Length; layer++)
					{
						List<Node> hNodes = new List<Node>();

						for (var n = 0; n < m_hidden[layer]; n++)
						{
							hNodes.Add(new HiddenNode(prevNodes, m_rand, m_weights[wIdx++]));
						}

						prevNodes = hNodes.Count + 1;
						m_layers.Add(hNodes);
					}

					// add the output layer
					List<Node> oNodes = new List<Node>();
					for (var n = 0; n < m_outputLabels.Count; n++)
					{
						var labelValueCount = m_outputLabels[n].valueCount;

						if (labelValueCount < 2)
						{
							// continuous
							oNodes.Add(new OutputNode(prevNodes, true, n, -1, m_rand, m_weights[wIdx++]));
						}
						else
						{
							oNodes.Add(new OutputNode(prevNodes, false, m_outputLabels[n].label, m_outputLabels[n].value, m_rand, m_weights[wIdx++]));
						}
					}

					m_layers.Add(oNodes);
				}
			}

			return (weights != null) && (weights.Count > 0);
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
				// add the input nodes
				List<Node> iNodes = new List<Node>();
				for (var i = 0; i < features.Cols(); i++)
				{
					iNodes.Add(new InputNode(i, 0, colMin[i], colMax[i], m_rand));
				}

				m_layers.Add(iNodes);

				int prevNodes = iNodes.Count + 1;
				int wIdx = 0;							// index into the weights array

				// add the hidden nodes
				for (var layer = 0; layer < m_hidden.Length; layer++)
				{
					// add the nodes for this layer
					List<Node> hNodes = new List<Node>();

					// if not the last 2 hidden layers, add c bias weight
					if (layer < m_hidden.Length - 2)
					{
						prevNodes++;
					}

					for (var n = 0; n < m_hidden[layer]; n++)
					{
						if (m_weights != null)
						{
							hNodes.Add(new HiddenNode(prevNodes, m_rand, m_weights[wIdx++]));
						}
						else
						{
							hNodes.Add(new HiddenNode(prevNodes, m_rand, null));
						}
					}

					prevNodes = hNodes.Count + 1;
					m_layers.Add(hNodes);
				}

				// add the output nodes - figure out how many outputs we need
				List<Node> oNodes = new List<Node>();
				for (var col = 0; col < labels.Cols(); col++)
				{
					var labelValueCount = labels.ValueCount(col);

					if (labelValueCount < 2)
					{
						// continuous
						if (m_weights != null)
						{
							oNodes.Add(new OutputNode(prevNodes, true, col, -1, m_rand, m_weights[wIdx++]));
						}
						else
						{
							oNodes.Add(new OutputNode(prevNodes, true, col, -1, m_rand, null));
						}
					}
					else
					{
						for (var n = 0; n < labelValueCount; n++)
						{
							if (m_weights != null)
							{
								oNodes.Add(new OutputNode(prevNodes, false, col, n, m_rand, m_weights[wIdx++]));
							}
							else
							{
								oNodes.Add(new OutputNode(prevNodes, false, col, n, m_rand, null));
							}
						}
					}
				}

				m_layers.Add(oNodes);
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
			int bestEpoch = 0;						// epoch number of best MSE
			int eCount;								// number of epochs since the best MSE
			bool checkDone;							// if true, check to see if we're done
			double bestTrainMSE = double.MaxValue;	// best training MSE so far
			double bestMSE = double.MaxValue;		// best validation MSE so far
			double bestAccuracy = double.MaxValue;	// best validationa accuracy so far

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

			if (m_weights != null)
			{
				// not training
				double trainMSE = VGetMSE(trainFeatures, trainLabels);
				epoch = 1;
				double mse = VGetMSE(validationFeatures, validationLabels);
				double accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);
				Console.WriteLine(string.Format("{0}\t{1}\t{2}\t{3}", epoch, trainMSE, mse, accuracy));
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine(string.Format("{0}\t{1}\t{2}\t{3}", epoch, trainMSE, mse, accuracy));
				}
			}
			else
			{
				for (int hLayer = 1; hLayer <= m_hidden.Length; hLayer++)
				{
					if (hLayer < m_hidden.Length)
					{
						// dbn layer
						epoch = 0;
						eCount = 0;
						checkDone = false;
						double wDelta = 0;
						double lastDelta = 0;
						double bestDelta = double.MaxValue;
						int maxEpochs = 500000 / trainFeatures.Rows();
						if (maxEpochs < 10)
						{
							maxEpochs = 10;
						}

						for (; ;)
						{
							// shuffle the training set
							trainFeatures.Shuffle(m_rand, trainLabels);
							wDelta = TrainDBN(hLayer, ++epoch, trainFeatures, trainLabels);

							Console.WriteLine(string.Format("{0}\t{1}", epoch, wDelta));
							if (m_outputFile != null)
							{
								m_outputFile.WriteLine(string.Format("{0}\t{1}", epoch, wDelta));
							}

							if (epoch > maxEpochs)
							{
								break;
							}
							else if (epoch == 1)
							{
								bestDelta = wDelta;
							}
							else if ((wDelta / lastDelta) >= 0.99)
							{
								if (!checkDone)
								{
									checkDone = true;
									eCount = 0;
								}
							}
							else if (wDelta < bestDelta)
							{
								checkDone = false;
							}
							else if (!checkDone)
							{
								checkDone = true;
								eCount = 0;
							}

							if (checkDone)
							{
								// check to see if we're done
								eCount++;
								if (eCount >= 5)
								{
									break;
								}
							}

							if (wDelta < bestDelta)
							{
								bestDelta = wDelta;
							}
							lastDelta = wDelta;
						}
					}
					else
					{
						// final hidden layer
						epoch = 0;
						eCount = 0;
						checkDone = false;

						double initialMSE = double.MaxValue;	// MSE for first epoch

						for (; ;)
						{
							// shuffle the training set
							trainFeatures.Shuffle(m_rand, trainLabels);
							double trainMSE = TrainEpoch(++epoch, trainFeatures, trainLabels);

							// check the MSE after this epoch
							double mse = VGetMSE(validationFeatures, validationLabels);

							// check the validation accuracy after this epoch
							double accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);

							Console.WriteLine(string.Format("{0}\t{1}\t{2}\t{3}", epoch, trainMSE, mse, accuracy));
							if (m_outputFile != null)
							{
								m_outputFile.WriteLine(string.Format("{0}\t{1}\t{2}\t{3}", epoch, trainMSE, mse, accuracy));
							}

							if (mse == 0.0)
							{
								// can't get better than this
								break;
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
									break;
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
								break;
							}
						}
					}
				}
			}

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
					foreach (OutputNode node in m_layers[m_layers.Count - 1])
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

		private void InitNodes()
		{
			for (var layer = 0; layer < m_layers.Count; layer++)
			{
				for (int idx = 0; idx < m_layers[layer].Count; idx++)
				{
					m_layers[layer][idx].index = idx;
				}
			}
		}

		private void DropNodes(bool keepAll = false)
		{
			for (var layer = 0; layer < m_layers.Count - 1; layer++)
			{
				foreach (var node in m_layers[layer])
				{
					if (layer == 0)
					{
						// input layer
						node.isActive = keepAll || (m_rand.NextDouble() <= m_pi);
					}
					else
					{
						// hidden layer
						node.isActive = keepAll || (m_rand.NextDouble() <= m_ph);
					}
				}
			}
		}

		private double TrainDBN(int hLayer, int epoch, VMatrix features, VMatrix labels)
		{
			double sse;
			double sseAccum = 0;
			object lo = new object();

			Console.Write(string.Format("TrainDBN {0} - ", hLayer));
			int cl = Console.CursorLeft;

//			if ((hLayer == 1) && (epoch == 1))
//			{
//				m_layers[0][0].weights[0] = 0;
//				m_layers[0][1].weights[0] = 0;
//				m_layers[1][0].weights[0] = 0.6;
//				m_layers[1][0].weights[1] = 0.4;
//				m_layers[1][0].weights[2] = 0;
//				//m_layers[1][0].weights[3] = 0;
//				m_layers[1][1].weights[0] = 0.5;
//				m_layers[1][1].weights[1] = -0.1;
//				m_layers[1][1].weights[2] = 0;
//				//m_layers[1][1].weights[3] = 0;
//			}

			for (var row = 0; row < features.Rows(); row++)
			{
				sse = 0;

				Console.SetCursorPosition(cl, Console.CursorTop);
				Console.Write(row);

				//DropNodes();

				// calculate the output
				for (var layer = 0; layer <= hLayer; layer++)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						node.net = 0;
						node.output = 0;
						node.sample = 0;
						node.net2 = 0;
						node.output2 = 0;
						node.sample2 = 0;
						node.error = 0;

						if (node.isActive)
						{
							if (layer == 0)
							{
								// input node
								node.output = features.Get(row, node.index);
								node.sample = node.output;
							}
							else
							{
								// calculate the net value
								int wCount = m_layers[layer - 1].Count;
								for (var w = 0; w < wCount; w++)
								{
									var nNode = m_layers[layer - 1][w];
									if (nNode.isActive)
									{
										if (m_sample)
										{
											node.net += node.weights[w] * nNode.sample;
										}
										else
										{
											node.net += node.weights[w] * nNode.output;
										}
									}
								}
								// add the bias
								node.net += node.weights[wCount];

								// calculate the output
								node.output = 1.0 / (1.0 + Math.Exp(-node.net));

								// sample
								lock (lo) { node.sample = (m_rand.NextDouble() < node.output ? 1 : 0); }
							}
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate P and x2 vectors
#if parallel
				Parallel.ForEach(m_layers[hLayer - 1], node =>
#else
				foreach (var node in m_layers[hLayer - 1])
#endif
				{
					if (node.isActive)
					{
						// calculate the net2 value
						for (var h = 0; h < m_layers[hLayer].Count; h++)
						{
							var hNode = m_layers[hLayer][h];
							if (hNode.isActive)
							{
								node.net2 += hNode.weights[node.index] * hNode.sample;
							}
						}
						// add the c bias
						node.net2 += node.weights[node.weights.Length - 1];

						// calculate the output
						node.output2 = 1.0 / (1.0 + Math.Exp(-node.net2));

						// sample
						lock (lo) { node.sample2 = (m_rand.NextDouble() < node.output2 ? 1 : 0); }
						
						double error = node.output2 - node.output;
						lock (lo) { sse += error * error; }
					}
#if parallel
				});
#else
				}
#endif

				sse /= m_layers[hLayer - 1].Count;

				// calculate Q2 and h2 vectors
#if parallel
				Parallel.ForEach(m_layers[hLayer], node =>
#else
				foreach (var node in m_layers[hLayer])
#endif
				{
					if (node.isActive)
					{
						// calculate the net2 value
						int wCount = m_layers[hLayer - 1].Count;
						for (var w = 0; w < wCount; w++)
						{
							var nNode = m_layers[hLayer - 1][w];
							if (nNode.isActive)
							{
								node.net2 += node.weights[w] * nNode.sample2;
							}
						}
						// add the bias
						node.net2 += node.weights[wCount];

						// calculate the output
						node.output2 = 1.0 / (1.0 + Math.Exp(-node.net2));

						// sample
						lock (lo) { node.sample2 = (m_rand.NextDouble() < node.output2 ? 1 : 0); }
					}
#if parallel
				});
#else
				}
#endif

				// calculate the weight changes and update the weights
#if parallel
				Parallel.ForEach(m_layers[hLayer], node =>
#else
				foreach (var node in m_layers[hLayer])
#endif
				{
					if (node.isActive)
					{
						// calculate the weight changes
						double delta;
						int wCount = m_layers[hLayer - 1].Count;
						for (var w = 0; w < wCount; w++)
						{
							if (node.isActive)
							{
								var dNode = m_layers[hLayer - 1][w];
								if (dNode.isActive)
								{
									delta = m_rate * (node.sample * dNode.output - node.output2 * dNode.sample2);
									//delta = m_rate * (node.sample * dNode.sample - node.sample2 * dNode.sample2);
									delta += m_momentum * node.deltas[w];
									node.deltas[w] = delta;
									node.weights[w] += delta;
								}
							}
						}

						// calculate the c bias weight change
						delta = m_rate * (node.sample - node.output2);
						delta += m_momentum * node.deltas[wCount];
						node.deltas[wCount] = delta;
						node.weights[wCount] += delta;
					}
#if parallel
				});
#else
				}
#endif

				// calculate the b bias weight changes and update the weights
#if parallel
				Parallel.ForEach(m_layers[hLayer - 1], node =>
#else
				foreach (var node in m_layers[hLayer - 1])
#endif
				{
					if (node.isActive)
					{
						// calculate the b bias weight change
						int wCount = node.weights.Length;
						double delta = m_rate * (node.output - node.sample2);
						delta += m_momentum * node.deltas[wCount - 1];
						node.deltas[wCount - 1] = delta;
						node.weights[wCount - 1] += delta;
					}
#if parallel
				});
#else
				}
#endif

				sseAccum += sse;
			}

			Console.WriteLine();

			return sseAccum / features.Rows();
		}

		private double TrainEpoch(int epoch, VMatrix features, VMatrix labels)
		{
			double sse = 0;
			object lo = new object();

			Console.Write("TrainEpoch ");
			int cl = Console.CursorLeft;

			StreamWriter aFile = null;
			if (epoch == 1)
			{
				aFile = File.CreateText("dbnTrain.arff");
				aFile.WriteLine("@RELATION DBN");
				aFile.WriteLine();
				for (var i = 1; i <= m_layers[m_layers.Count - 3].Count; i++)
				{
					aFile.WriteLine(string.Format("@ATTRIBUTE hn{0}	real", i));
				}
				aFile.WriteLine("@ATTRIBUTE class	{0,1,2,3,4,5,6,7,8,9}");
				aFile.WriteLine();
				aFile.WriteLine("@DATA");
			}

			for (var row = 0; row < features.Rows(); row++)
			{
				Console.SetCursorPosition(cl, Console.CursorTop);
				Console.Write(row);

				//DropNodes();

				// calculate the output
				for (var layer = 0; layer < m_layers.Count; layer++)
				{
					Parallel.ForEach(m_layers[layer], node =>
					{
						node.net = 0;
						node.output = 0;
						node.error = 0;

						if (node.isActive)
						{
							if (layer == 0)
							{
								// input node
								node.output = features.Get(row, node.index);
							}
							else
							{
								// calculate the net value
								int wCount = m_layers[layer - 1].Count;
								for (var w = 0; w < wCount; w++)
								{
									var nNode = m_layers[layer - 1][w];
									if (nNode.isActive)
									{
										node.net += node.weights[w] * nNode.output;
									}
								}
								// add the bias
								node.net += node.weights[wCount];

								// calculate the output
								node.output = 1.0 / (1.0 + Math.Exp(-node.net));

								// if this is a dbn node, check to see if we're sampling
								if (m_sample && (layer < m_layers.Count - 2))
								{
									lock (lo) { node.sample = (m_rand.NextDouble() < node.output ? 1 : 0); }
									node.output = node.sample;
								}
							}
						}
					});
				}

				if ((epoch == 1) && (aFile != null))
				{
					StringBuilder asb = new StringBuilder();
					foreach (var node in m_layers[m_layers.Count - 3])
					{
						if (m_sample)
						{
							asb.Append(node.sample);
						}
						else
						{
							asb.Append(node.output);
						}
						asb.Append(',');
					}

					OutputNode oNode = m_layers[m_layers.Count - 1][0] as OutputNode;
					double ov = labels.Get(row, oNode.labelCol);
					string on = labels.AttrValue(oNode.labelCol, (int)ov);
					asb.Append(on);

					aFile.WriteLine(asb.ToString());
				}

				// calculate the error and weight changes
				for (var layer = m_layers.Count - 1; layer > m_layers.Count - 3; layer--)
				{
					Parallel.ForEach(m_layers[layer], node =>
					{
						if (node.isActive)
						{
							double fPrime = node.output * (1.0 - node.output);
							if (layer == m_layers.Count - 1)
							{
								// output layer
								OutputNode oNode = node as OutputNode;
								double target = labels.Get(row, oNode.labelCol);
								if (!oNode.isContinuous)
								{
									// nominal
									if (target == oNode.labelVal)
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
									if (tn.isActive)
									{
										sum += tn.error * tn.weights[node.index];
									}
								}
								node.error = sum * fPrime;
							}

							// calculate the weight changes
							double delta;
							int wCount = m_layers[layer - 1].Count;
							for (var w = 0; w < wCount; w++)
							{
								if (node.isActive)
								{
									var dNode = m_layers[layer - 1][w];
									if (dNode.isActive)
									{
										delta = m_rate * node.error * dNode.output;
										delta += m_momentum * node.deltas[w];
										node.deltas[w] = delta;
									}
								}
							}

							// calculate the bias weight change
							delta = m_rate * node.error;
							delta += m_momentum * node.deltas[wCount];
							node.deltas[wCount] = delta;
						}
					});
				}

				// update the weights
				for (var layer = m_layers.Count - 2; layer < m_layers.Count; layer++)
				{
					Parallel.ForEach(m_layers[layer], node =>
					{
						if (node.isActive)
						{
							int wCount = m_layers[layer - 1].Count;
							for (var w = 0; w < wCount; w++)
							{
								var wNode = m_layers[layer - 1][w];
								if (wNode.isActive)
								{
									node.weights[w] += node.deltas[w];
								}

								// update the bias weight
								node.weights[wCount] += node.deltas[wCount];
							}
						}
					});
				}
			}

			Console.WriteLine();

			if ((epoch == 1) && (aFile != null))
			{
				aFile.Close();
			}

			return sse / features.Rows();
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
			object lo = new object();

			Console.Write("VGetMSE ");
			int cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				Console.SetCursorPosition(cl, Console.CursorTop);
				Console.Write(row);

				// calculate the output
				for (var layer = 0; layer < m_layers.Count; layer++)
				{
					Parallel.ForEach(m_layers[layer], node =>
					{
						node.net = 0;
						node.output = 0;

						if (layer == 0)
						{
							// input node
							node.output = features.Get(row, node.index);
						}
						else
						{
							// calculate the net value
							int wCount = m_layers[layer - 1].Count;
							for (var w = 0; w < wCount; w++)
							{
								double weight = node.weights[w];
								if (layer == 1)
								{
									//weight *= m_pi;
								}
								else
								{
									//weight *= m_ph;
								}
								node.net += weight * m_layers[layer - 1][w].output;
							}
							// add the bias
							node.net += node.weights[wCount];

							node.output = 1.0 / (1.0 + Math.Exp(-node.net));
							
							// if this is a dbn node, check to see if we're sampling
							if (m_sample && (layer < m_layers.Count - 2))
							{
								lock (lo) { node.sample = (m_rand.NextDouble() < node.output ? 1 : 0); }
								node.output = node.sample;
							}
						}
					});
				}

				// calculate the error of the output layer
				for (var n = 0; n < m_layers[m_layers.Count - 1].Count; n++)
				{
					OutputNode node = m_layers[m_layers.Count - 1][n] as OutputNode;
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
			for (var layer = 1; layer < m_layers.Count; layer++)
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
			object lo = new object();

			for (var layer = 0; layer < m_layers.Count; layer++)
			{
				Parallel.ForEach(m_layers[layer], node =>
				{
					node.net = 0;
					node.output = 0;

					if (layer == 0)
					{
						// input node
						node.output = features[node.index];
					}
					else
					{
						// calculate the net value
						int wCount = m_layers[layer - 1].Count;
						for (var w = 0; w < wCount; w++)
						{
							double weight = node.weights[w];
							if (layer == 1)
							{
								//weight *= m_pi;
							}
							else
							{
								//weight *= m_ph;
							}
							node.net += weight * m_layers[layer - 1][w].output;
						}
						// add the bias
						node.net += node.weights[wCount];

						node.output = 1.0 / (1.0 + Math.Exp(-node.net));

						// if this is a dbn node, check to see if we're sampling
						if (m_sample && (layer < m_layers.Count - 2))
						{
							lock (lo) { node.sample = (m_rand.NextDouble() < node.output ? 1 : 0); }
							node.output = node.sample;
						}
					}
				});
			}

			int labelIdx = 0;
			for (var n = 0; n < m_layers[m_layers.Count - 1].Count; n++)
			{
				OutputNode node = m_layers[m_layers.Count - 1][n] as OutputNode;

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
						OutputNode tn = m_layers[m_layers.Count - 1][n + nIdx] as OutputNode;
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
