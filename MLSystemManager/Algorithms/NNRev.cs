﻿//#define parallel

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	/// <summary>
	/// Reversed Neural Net - old
	/// </summary>
	class NNRev : SupervisedLearner
	{
		private List<List<Node>> m_layers;
		private Random m_rand;
		private double m_rate = 0.1;
		private double m_momentum = 0.9;
		private string m_activation = "sigmoid";
		private double m_actLeak = 0;
		private double m_actThreshold = 0;
		private double m_actSlope = 1;
		private bool m_actRandom = false;
		private int[] m_hidden = null;
		private StreamWriter m_outputFile = null;

		public class Node
		{
			public int index { get; set; }					// the index of this node
			public double threshold { get; set; }			// relu threshold
			public double net { get; set; }					// the net for this node
			public double output { get; set; }				// the output of this node
			public double error { get; set; }				// the error for this node
			public double target { get; set; }				// the target for this node
			public double tNet { get; set; }				// the t net for this node
			public double tMult { get; set; }				// the t multiplier for this node
			public double[] weights { get; set; }			// the weights for all nodes connected to this node
			public double[] bestWeights { get; set; }		// the best weights so far
			public double[] deltas { get; set; }			// weight deltas from previous epoch

			public Node()
			{
				index = 0;
				net = 0;
				output = 0;
				error = 0;
				target = 0;
				tNet = 0;
				tMult = 0;
				weights = null;
				bestWeights = null;
				deltas = null;
			}

			public Node(int numWeights, Random rand)
			{
				if (numWeights > 0)
				{
					weights = new double[numWeights];
					bestWeights = new double[numWeights];
					for (var i = 0; i < numWeights; i++)
					{
						weights[i] = 0.1 - (rand.NextDouble() * 0.2);
						bestWeights[i] = weights[i];
					}

					deltas = new double[numWeights];
					for (var i = 0; i < numWeights; i++)
					{
						deltas[i] = 0;
					}
				}
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

			public void InitDeltas()
			{
				if (deltas != null)
				{
					for (var i = 0; i < deltas.Length; i++)
					{
						deltas[i] = 0;
					}
				}
			}
		}

		public class InputNode : Node
		{
			public int feature { get; set; }
			public int valueCount { get; set; }
			public InputNode(int feature, int valueCount, Random rand)
				: base(0, rand)
			{
				this.feature = feature;
				this.valueCount = valueCount;
			}
		}

		public class HiddenNode : Node
		{
			public HiddenNode(int numWeights, Random rand)
				: base(numWeights, rand)
			{
			}
		}

		public class OutputNode : Node
		{
			public bool isContinuous { get; set; }			// true if the outout is continuous
			public int labelCol { get; set; }				// the label column that this output node corresponds to
			public double labelVal { get; set; }			// the value of the label column that this output node corresponds to
			public OutputNode(int numWeights, bool isContinuous, int labelCol, double labelVal, Random rand)
				: base(numWeights, rand)
			{
				this.isContinuous = isContinuous;
				this.labelCol = labelCol;
				this.labelVal = labelVal;
			}
		}

		public NNRev()
		{
			m_rand = new Random();
			m_layers = new List<List<Node>>();
		}

		public NNRev(Parameters parameters)
		{
			m_rand = Rand.Get();
			m_rate = parameters.Rate;
			m_momentum = parameters.Momentum;
			m_activation = parameters.Activation;
			var sa = parameters.ActParameter.Split(',');
			if (sa.Length > 0)
			{
				m_actLeak = double.Parse(sa[0]);
			}
			if (sa.Length > 1)
			{
				m_actThreshold = double.Parse(sa[1]);
			}
			if (sa.Length > 2)
			{
				m_actSlope = double.Parse(sa[2]);
			}
			if (sa.Length > 3)
			{
				m_actRandom = (sa[3].ToLower() == "r");
			}

			m_hidden = parameters.Hidden;
			m_layers = new List<List<Node>>();
		}

		public override void Train(Matrix features, Matrix labels)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels)
		{
			if (m_hidden.Length < 1)
			{
				m_hidden = new int[1] { features.Cols() * 2 };
			}

			// add the input nodes
			List<Node> iNodes = new List<Node>();
			for (var i = 0; i < features.Cols(); i++)
			{
				iNodes.Add(new InputNode(i, 0, m_rand));
			}

			m_layers.Add(iNodes);
			int prevNodes = iNodes.Count + 1;

			// add the hidden nodes
			for (int h = 0; h < m_hidden.Length; h++)
			{
				List<Node> hNodes = new List<Node>();

				for (var n = 0; n < m_hidden[h]; n++)
				{
					hNodes.Add(new HiddenNode(prevNodes, m_rand));
				}

				m_layers.Add(hNodes);

				prevNodes = hNodes.Count + 1;
			}

			// add output nodes
			List<Node> oNodes = new List<Node>();

			// figure out how many outputs we need
			for (var col = 0; col < labels.Cols(); col++)
			{
				var labelValueCount = labels.ValueCount(col);

				if (labelValueCount < 2)
				{
					// continuous
					oNodes.Add(new OutputNode(prevNodes, true, col, -1, m_rand));
				}
				else
				{
					for (var n = 0; n < labelValueCount; n++)
					{
						oNodes.Add(new OutputNode(prevNodes, false, col, n, m_rand));
					}
				}
			}

			m_layers.Add(oNodes);

			InitNodes();

			int trainSize = (int)(0.75 * features.Rows());
			VMatrix trainFeatures = new VMatrix(features, 0, 0, trainSize, features.Cols());
			VMatrix trainLabels = new VMatrix(labels, 0, 0, trainSize, labels.Cols());
			VMatrix validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			VMatrix validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			Console.Write("Layers: ");
			Console.Write(features.Cols());
			Console.Write('x');
			for (var l = 0; l < m_hidden.Length; l++)
			{
				Console.Write(m_hidden[l]);
				Console.Write('x');
			}
			Console.WriteLine(oNodes.Count);
			Console.WriteLine("Momentum: " + m_momentum);
			Console.WriteLine("AF: " + m_activation);
			Console.WriteLine(string.Format("AParam: {0},{1},{2},{3}", m_actLeak, m_actThreshold, m_actSlope, m_actRandom));
			Console.WriteLine("R-E-C\tMSE (validation)");
			if (m_outputFile != null)
			{
				m_outputFile.Write("Layers: ");
				m_outputFile.Write(features.Cols());
				m_outputFile.Write('x');
				for (var l = 0; l < m_hidden.Length; l++)
				{
					m_outputFile.Write(m_hidden[l]);
					m_outputFile.Write('x');
				}
				m_outputFile.WriteLine(oNodes.Count);
				m_outputFile.WriteLine("Momentum: " + m_momentum);
				m_outputFile.WriteLine("AF: " + m_activation);
				m_outputFile.WriteLine(string.Format("AParam: {0},{1},{2},{3}", m_actLeak, m_actThreshold, m_actSlope, m_actRandom));
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
				m_outputFile.WriteLine("R-E-C\tMSE (validation)");
			}

			for (int round = 1; round < m_layers.Count; round++)
			{
				int epoch = 0;							// current epoch number
				int bestEpoch = 0;						// epoch number of best MSE
				int eCount = 0;							// number of epochs since the best MSE
				bool checkDone = false;					// if true, check to see if we're done
				double initialMSE = double.MaxValue;	// MSE for first epoch
				double bestMSE = double.MaxValue;		// best validation MSE so far

				for (; ; )
				{
					// shuffle the training set
					trainFeatures.Shuffle(m_rand, trainLabels);
					TrainEpoch(++epoch, trainFeatures, trainLabels, m_layers.Count - round);

					// check the MSE after this epoch
					double mse = VGetMSE(validationFeatures, validationLabels);

					Console.WriteLine(string.Format("{0}-{1}-{2}\t{3}", round, epoch, eCount, mse));
					if (m_outputFile != null)
					{
						m_outputFile.WriteLine(string.Format("{0}-{1}-{2}\t{3}", round, epoch, eCount, mse));
						m_outputFile.Flush();
					}

					if ((mse == 0.0) || (epoch > 10000))
					{
						break;
					}
					else if ((epoch == 1) || (mse < bestMSE))
					{
						if (epoch == 1)
						{
							// save the initial MSE
							initialMSE = mse;
						}
						else if (!checkDone && (mse < initialMSE * 0.75))
						{
							checkDone = true;
						}
						eCount = 0;

						// save the best for later
						bestMSE = mse;
						bestEpoch = epoch;
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
				}

				for (var layer = 0; layer < m_layers.Count - 1; layer++)
				{
					foreach (var node in m_layers[layer])
					{
						node.RestoreBestWeights();
						node.InitDeltas();
					}
				}

				Console.WriteLine(string.Format("Best Weights (from Epoch {0}, valMSE={1})", bestEpoch, bestMSE));
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine();
					m_outputFile.WriteLine(string.Format("Best Weights (from Epoch {0}, valMSE={1})", bestEpoch, bestMSE));
					m_outputFile.Flush();
				}
			}

			if (m_outputFile != null)
			{
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
			}

			if (m_outputFile != null)
			{
				m_outputFile.Close();
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

		private void TrainEpoch(int epoch, VMatrix features, VMatrix labels, int currLayer)
		{
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
						node.output = 0;
						node.error = 0;
						node.target = 0;
						node.tNet = 0;
						node.tMult = 0;

						if (layer == 0)
						{
							// input node
							node.output = features.Get(row, node.index);
						}
						else
						{
							// calculate the net value
							for (var w = 0; w < node.weights.Length - 1; w++)
							{
								node.net += node.weights[w] * m_layers[layer - 1][w].output;
							}
							// add the bias
							node.net += node.weights[node.weights.Length - 1];

							// calculate the output
							if (m_activation == "relu")
							{
								node.output = (node.net < node.threshold ? ((node.net - node.threshold) * m_actLeak) + node.threshold : node.net * m_actSlope);
							}
							else if (m_activation == "softsign")
							{
								node.output = (node.net / (1.0 + Math.Abs(node.net)));
							}
							else if (m_activation == "softplus")
							{
								node.output = Math.Log(1.0 + Math.Exp(node.net));
							}
							else
							{
								node.output = 1.0 / (1.0 + Math.Exp(-node.net));
							}
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error and weight changes
				int minLayer = 1;// currLayer;
				for (var layer = m_layers.Count - 1; layer >= minLayer; layer--)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						double fPrime;
						if (m_activation == "relu")
						{
							fPrime = (node.output < node.threshold ? m_actLeak : m_actSlope);
						}
						else if (m_activation == "softsign")
						{
							fPrime = 1.0 / ((1.0 - Math.Abs(node.output)) * (1.0 - Math.Abs(node.output)));
						}
						else if (m_activation == "softplus")
						{
							fPrime = 1.0 / (1.0 + Math.Exp(-node.output));
						}
						else
						{
							fPrime = node.output * (1.0 - node.output);
						}

						if (layer == m_layers.Count - 1)
						{
							// output layer
							OutputNode oNode = node as OutputNode;
							node.target = labels.Get(row, oNode.labelCol);

							if (!oNode.isContinuous)
							{
								// nominal
								if (node.target == oNode.labelVal)
								{
									node.target = 0.9;
								}
								else
								{
									node.target = 0.1;
								}
							}

							var error = node.target - node.output;
							node.error = error * fPrime;

							node.tNet = -Math.Log((1.0 / node.target) - 1);
							node.tMult = ((node.tNet - node.weights[node.weights.Length - 1]) / (node.net - node.weights[node.weights.Length - 1]));
						}
						else
						{
							// hidden layer
							double sum = 0;
							foreach (var tn in m_layers[layer + 1])
							{
								sum += tn.error * tn.weights[node.index];
								node.tMult += tn.tMult;
							}
							node.error = sum * fPrime;

							node.tMult /= m_layers[layer + 1].Count;
							node.target = node.output * node.tMult;
							node.tNet = -Math.Log((1.0 / node.target) - 1);
							node.tMult = ((node.tNet - node.weights[node.weights.Length - 1]) / (node.net - node.weights[node.weights.Length - 1]));
						}

						// calculate the weight changes
						double delta;
						for (var w = 0; w < node.weights.Length - 1; w++)
						{
							delta = m_rate * node.error * m_layers[layer - 1][w].output;
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
				minLayer = 1;// currLayer;
				for (var layer = minLayer; layer < m_layers.Count; layer++)
				{
#if parallel
					Parallel.ForEach(m_layers[layer], node =>
#else
					foreach (var node in m_layers[layer])
#endif
					{
						for (var w = 0; w < node.weights.Length; w++)
						{
							node.weights[w] += node.deltas[w];
						}
#if parallel
					});
#else
					}
#endif
				}
			}

			Console.WriteLine();
		}

		private double mActivation(double net)
		{
			//return (net < 0 ? 0.01 * net : net);
			return 1.0 / (1.0 + Math.Exp(-net));
		}

		private double mFPrime(double net)
		{
			//return (net < 0 ? 0.01 : 1.0);
			return net * (1.0 - net);
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
				if (((row % 10) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

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
							for (var w = 0; w < node.weights.Length - 1; w++)
							{
								node.net += node.weights[w] * m_layers[layer - 1][w].output;
							}
							// add the bias
							node.net += node.weights[node.weights.Length - 1];

							// calculate the output
							if (m_activation == "relu")
							{
								node.output = (node.net < node.threshold ? ((node.net - node.threshold) * m_actLeak) + node.threshold : node.net * m_actSlope);
							}
							else if (m_activation == "softsign")
							{
								node.output = (node.net / (1.0 + Math.Abs(node.net)));
							}
							else if (m_activation == "softplus")
							{
								node.output = Math.Log(1.0 + Math.Exp(node.net));
							}
							else
							{
								node.output = 1.0 / (1.0 + Math.Exp(-node.net));
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
			m_outputFile.Flush();
		}

		public override void Predict(double[] features, double[] labels)
		{
			object lo = new object();

			for (var layer = 0; layer < m_layers.Count; layer++)
			{
				Parallel.ForEach(m_layers[layer], node =>
				{
					node.net = 0;

					if (layer == 0)
					{
						// input node
						node.output = features[node.index];
					}
					else
					{
						// calculate the net value
						for (var w = 0; w < node.weights.Length - 1; w++)
						{
							node.net += node.weights[w] * m_layers[layer - 1][w].output;
						}
						// add the bias
						node.net += node.weights[node.weights.Length - 1];

						// calculate the output
						if (m_activation == "relu")
						{
							node.output = (node.net < node.threshold ? ((node.net - node.threshold) * m_actLeak) + node.threshold : node.net * m_actSlope);
						}
						else if (m_activation == "softsign")
						{
							node.output = (node.net / (1.0 + Math.Abs(node.net)));
						}
						else if (m_activation == "softplus")
						{
							node.output = Math.Log(1.0 + Math.Exp(node.net));
						}
						else
						{
							node.output = 1.0 / (1.0 + Math.Exp(-node.net));
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
