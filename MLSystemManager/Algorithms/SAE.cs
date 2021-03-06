﻿#define parallel

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	/// <summary>
	/// Stacked Autoencoder
	/// </summary>
	class SAE : SupervisedLearner
	{
		private List<List<Node>> m_layers;
		private Random m_rand;
		private double m_rate = 0.1;
		private double m_momentum = 0.9;
		private double m_corruptLevel = 0.15;
		private int[] m_hidden = null;
		private List<double[]> m_weights = null;
		private List<OutputLabel> m_outputLabels = null;
		private StreamWriter m_outputFile = null;

		public class Node
		{
			public int index { get; set; }					// the index of this node
			public double net { get; set; }					// the net for this node
			public double output { get; set; }				// the output of this node
			public double output2 { get; set; }				// the uncorrupted output of this node
			public double error { get; set; }				// the error for this node
			public double[] weights { get; set; }			// the weights for all nodes connected to this node
			public double[] bestWeights { get; set; }		// the best weights so far
			public double[] deltas { get; set; }			// weight deltas from previous epoch

			public Node()
			{
				index = 0;
				net = 0;
				output = 0;
				output2 = 0;
				error = 0;
				weights = null;
				bestWeights = null;
				deltas = null;
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
			public InputNode(int feature, int valueCount, Random rand)
				: base(0, rand, null)
			{
				this.feature = feature;
				this.valueCount = valueCount;
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

		public SAE()
		{
			m_rand = new Random();
			m_layers = new List<List<Node>>();
		}

		public SAE(Parameters parameters)
		{
			m_rand = Rand.Get();
			m_rate = parameters.Rate;
			m_momentum = parameters.Momentum;
			m_corruptLevel = parameters.CorruptLevel;
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

			int wIdx = 0;							// index into the weights array
			
			// add the input nodes
			List<Node> iNodes = new List<Node>();
			for (var i = 0; i < features.Cols(); i++)
			{
				iNodes.Add(new InputNode(i, 0, m_rand));
			}

			m_layers.Add(iNodes);

			int trainSize = (int)(0.75 * features.Rows());
			VMatrix trainFeatures = new VMatrix(features, 0, 0, trainSize, features.Cols());
			VMatrix trainLabels = new VMatrix(labels, 0, 0, trainSize, labels.Cols());
			VMatrix validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			VMatrix validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			Console.WriteLine("R-E-C\tMSE (training)\t\tMSE (validation)\taccuracy (validation)");
			if (m_outputFile != null)
			{
				m_outputFile.WriteLine("Momentum: " + m_momentum);
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
				m_outputFile.WriteLine("R-E-C\tMSE (training)\t\tMSE (validation)\taccuracy (validation)");
			}

			if (m_weights != null)
			{
				// not training
				double trainMSE = VGetMSE(trainFeatures, trainLabels);
				double mse = VGetMSE(validationFeatures, validationLabels);
				double accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);
				Console.WriteLine($"1\t{trainMSE}\t{mse}\t{accuracy}");
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine($"1\t{trainMSE}\t{mse}\t{accuracy}");
				}
			}
			else
			{
				for (int round = 1; round <= m_hidden.Length + 1; round++)
				{
					if (round <= m_hidden.Length)
					{
						// add hidden nodes
						int prevNodes = m_layers[m_layers.Count - 1].Count + 1;
						List<Node> hNodes = new List<Node>();

						for (var n = 0; n < m_hidden[m_layers.Count - 1]; n++)
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

						m_layers.Add(hNodes);

						prevNodes = hNodes.Count + 1;

						// add output nodes
						List<Node> oNodes = new List<Node>();

						if (round < m_hidden.Length)
						{
							// dae layer - add output nodes to match inputs
							for (var col = 0; col < m_layers[m_layers.Count - 2].Count; col++)
							{
								oNodes.Add(new OutputNode(prevNodes, true, col, -1, m_rand, null));
							}
						}
						else
						{
							// final layer - figure out how many outputs we need
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
						}

						m_layers.Add(oNodes);

						InitNodes();
					}

					int epoch = 0;							// current epoch number
					int bestEpoch = 0;						// epoch number of best MSE
					int eCount = 0;							// number of epochs since the best MSE
					bool checkDone = false;					// if true, check to see if we're done
					double mse = 0;							// validation MSE
					double bestTrainMSE = double.MaxValue;	// best training MSE so far
					double bestMSE = double.MaxValue;		// best validation MSE so far
					double accuracy = 0;					// validation accuracy
					double bestAccuracy = double.MaxValue;	// best validationa accuracy so far

					for (; ; )
					{
						// shuffle the training set
						trainFeatures.Shuffle(m_rand, trainLabels);
						double trainMSE = TrainEpoch(++epoch, trainFeatures, trainLabels, round < m_hidden.Length, round > m_hidden.Length);

						// check the MSE after this epoch
						if (round < m_hidden.Length)
						{
							mse = IGetMSE(validationFeatures);
							accuracy = 0;
						}
						else
						{
							mse = VGetMSE(validationFeatures, validationLabels);

							// check the validation accuracy
							accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);
						}

						Console.WriteLine($"{round}-{epoch}-{eCount}\t{trainMSE}\t{mse}\t{accuracy}");
						if (m_outputFile != null)
						{
							m_outputFile.WriteLine($"{round}-{epoch}-{eCount}\t{trainMSE}\t{mse}\t{accuracy}");
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
								bestMSE = mse;
							}
							else if ((mse / bestMSE) > 0.99)
							{
								if (!checkDone)
								{
									checkDone = true;
									eCount = 0;
								}
							}
							else
							{
								checkDone = false;
								eCount = 0;
							}

							// save the best for later
							bestTrainMSE = trainMSE;
							bestMSE = mse;
							bestAccuracy = accuracy;
							bestEpoch = epoch;
							for (var layer = 0; layer < m_layers.Count - 1; layer++)
							{
								foreach (var node in m_layers[layer])
								{
									node.SaveBestWeights();
								}
							}
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
							if (eCount >= 20)
							{
								break;
							}
						}
					}

					if (round < m_hidden.Length)
					{
						// delete the output layer
						m_layers.RemoveAt(m_layers.Count - 1);
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

						Console.WriteLine($"Best Weights (from Epoch {bestEpoch}, trainMSE={bestTrainMSE}, valMSE={bestMSE})");
						if (m_outputFile != null)
						{
							m_outputFile.WriteLine();
							m_outputFile.WriteLine($"Best Weights (from Epoch {bestEpoch}, trainMSE={bestTrainMSE}, valMSE={bestMSE})");
							m_outputFile.Flush();
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

		private double TrainEpoch(int epoch, VMatrix features, VMatrix labels, bool isDAE, bool trainAll)
		{
			double sse = 0;
			object lo = new object();

			Console.Write("TrainEpoch ");
			int cl = Console.CursorLeft;

			StreamWriter aFile = null;
			if (!isDAE && (epoch == 1))
			{
				aFile = File.CreateText("dbnTrain.arff");
				aFile.WriteLine("@RELATION DAE");
				aFile.WriteLine();
				for (var i = 1; i <= m_layers[m_layers.Count - 3].Count; i++)
				{
					aFile.WriteLine($"@ATTRIBUTE hn{i}	real");
				}
				aFile.WriteLine("@ATTRIBUTE class	{0,1,2,3,4,5,6,7,8,9}");
				aFile.WriteLine();
				aFile.WriteLine("@DATA");
			}

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
						node.output2 = 0;
						node.error = 0;

						if (layer == 0)
						{
							// input node
							node.output = features.Get(row, node.index);
							node.output2 = node.output;
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
							node.output = Activation(node.net);
							node.output2 = node.output;
						}

						if (isDAE && (layer == m_layers.Count - 3) && (node.output != 0))
							lock (lo)
							{
								// corrupt the output
								if (m_rand.NextDouble() < m_corruptLevel)
								{
									node.output = 0;
								}
							}
#if parallel
					});
#else
					}
#endif
				}

				if (!isDAE && (epoch == 1) && (aFile != null))
				{
					StringBuilder asb = new StringBuilder();
					foreach (var node in m_layers[m_layers.Count - 3])
					{
						asb.Append(node.output);
						asb.Append(',');
					}

					OutputNode oNode = m_layers[m_layers.Count - 1][0] as OutputNode;
					double ov = labels.Get(row, oNode.labelCol);
					string on = labels.AttrValue(oNode.labelCol, (int)ov);
					asb.Append(on);

					aFile.WriteLine(asb.ToString());
				}

				// calculate the error and weight changes
				int minLayer = (trainAll ? 0 : m_layers.Count - 3);
				for (var layer = m_layers.Count - 1; layer > minLayer; layer--)
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
							OutputNode oNode = node as OutputNode;
							double target;
							if (isDAE)
							{
								target = m_layers[m_layers.Count - 3][oNode.index].output2;
							}
							else
							{
								target = labels.Get(row, oNode.labelCol);
							}

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
								sum += tn.error * tn.weights[node.index];
							}
							node.error = sum * fPrime;
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
				minLayer = (trainAll ? 1 : m_layers.Count - 2);
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

				//if (isDAE && (m_outputFile != null))
				//{
				//	m_outputFile.WriteLine(string.Format("TrainEpoch {0} {1}", epoch, row));
				//	StringBuilder sb = new StringBuilder();
				//	foreach (var node in m_layers[m_layers.Count - 3])
				//	{
				//		if (sb.Length > 0)
				//		{
				//			sb.Append(',');
				//		}
				//		sb.Append(node.output2);
				//	}
				//	m_outputFile.WriteLine(sb.ToString());

				//	sb.Clear();
				//	foreach (var node in m_layers[m_layers.Count - 1])
				//	{
				//		if (sb.Length > 0)
				//		{
				//			sb.Append(',');
				//		}
				//		sb.Append(node.output);
				//	}
				//	m_outputFile.WriteLine(sb.ToString());
				//	m_outputFile.WriteLine();
				//	m_outputFile.Flush();
				//}
			}

			Console.WriteLine();

			if (!isDAE && (epoch == 1) && (aFile != null))
			{
				aFile.Close();
			}

			if (isDAE)
			{
				sse /= m_layers[m_layers.Count - 1].Count;
			}
			sse /= features.Rows();

			return sse;
		}

		private double Activation(double net)
		{
			return (net < 0 ? 0.01 * net : net);
			// return 1.0 / (1.0 + Math.Exp(-node.net));
		}

		private double FPrime(double net)
		{
			return (net < 0 ? 0.01 : 1.0);
			//return net * (1.0 - net);
		}

		private double IGetMSE(VMatrix features)
		{
			double sse = 0;
			object lo = new object();

			Console.Write("IGetMSE ");
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

							node.output = Activation(node.net);
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error of the output layer
#if parallel
				Parallel.ForEach(m_layers[m_layers.Count - 1], node =>
#else
				foreach (var node in m_layers[m_layers.Count - 1])
#endif
				{
					var error = m_layers[m_layers.Count - 3][node.index].output - node.output;

					// update the error
					lock (lo) { sse += error * error; }
#if parallel
				});
#else
				}
#endif

				//if (m_outputFile != null)
				//{
				//	m_outputFile.WriteLine("IGetMSE");
				//	StringBuilder sb = new StringBuilder();
				//	foreach (var node in m_layers[m_layers.Count - 3])
				//	{
				//		if (sb.Length > 0)
				//		{
				//			sb.Append(',');
				//		}
				//		sb.Append(node.output);
				//	}
				//	m_outputFile.WriteLine(sb.ToString());

				//	sb.Clear();
				//	foreach (var node in m_layers[m_layers.Count - 1])
				//	{
				//		if (sb.Length > 0)
				//		{
				//			sb.Append(',');
				//		}
				//		sb.Append(node.output);
				//	}
				//	m_outputFile.WriteLine(sb.ToString());
				//	m_outputFile.WriteLine();
				//	m_outputFile.Flush();
				//}
			}

			sse /= m_layers[m_layers.Count - 1].Count;
			sse /= features.Rows();

			Console.WriteLine();

			return sse;
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

							node.output = Activation(node.net);
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
						m_outputFile.Write($"{node.weights[w]}\t");
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

						node.output = Activation(node.net);
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
