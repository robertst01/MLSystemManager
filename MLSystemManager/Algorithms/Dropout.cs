﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	public class Dropout : SupervisedLearner
	{
		private List<List<Node>> m_layers;
		private Random m_rand;
		private double m_rate = 0.1;
		private double m_momentum = 0.9;
		private double m_pi = 0.8;
		private double m_ph = 0.5;
		private int[] m_hidden = null;
		private List<double[]> m_weights = null;
		private List<OutputLabel> m_outputLabels = null;
		private StreamWriter m_outputFile = null;

		public class Node
		{
			public int index { get; set; }					// the index of this node
			public double net { get; set; }					// the net for this node
			public double output { get; set; }				// the output of this node
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
			public InputNode(int feature, int valueCount, Random rand) : base(0, rand, null)
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

		public Dropout()
		{
			m_rand = new Random();
			m_layers = new List<List<Node>>();
		}

		public Dropout(Parameters parameters)
		{
			m_rand = Rand.Get();
			m_rate = parameters.Rate;
			m_momentum = parameters.Momentum;
			m_ph = parameters.Ph;
			m_pi = parameters.Pi;
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
			var iNodes = new List<Node>();
			for (var i = 0; i < features.Cols(); i++)
			{
				iNodes.Add(new InputNode(i, 0, m_rand));
			}

			m_layers.Add(iNodes);

			var prevNodes = iNodes.Count + 1;
			var wIdx = 0;							// index into the weights array

			// add the hidden nodes
			for (var layer = 0; layer < m_hidden.Length; layer++)
			{
				// add the nodes for this layer
				var hNodes = new List<Node>();

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
			var oNodes = new List<Node>();
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

			InitNodes();

			var trainSize = (int)(0.75 * features.Rows());
			var trainFeatures = new VMatrix(features, 0, 0, trainSize, features.Cols());
			var trainLabels = new VMatrix(labels, 0, 0, trainSize, labels.Cols());
			var validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			var validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			var epoch = 0;							// current epoch number
			var bestTrainMSE = double.MaxValue;	// best training MSE so far
			var bestMSE = double.MaxValue;		// best validation MSE so far
			var bestAccuracy = double.MaxValue;	// best validationa accuracy so far
			var initialMSE = double.MaxValue;	// MSE for first epoch
			var eCount = 0;							// number of epochs since the best MSE
			var bestEpoch = 0;						// epoch number of best MSE
			var done = false;
			var checkDone = false;					// if true, check to see if we're done

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
				var mse = VGetMSE(validationFeatures, validationLabels);

				// check the validation accuracy after this epoch
				var accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);

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
		}

		private void InitNodes()
		{
			for (var layer = 0; layer < m_layers.Count - 1; layer++)
			{
				for (var idx = 0; idx < m_layers[layer].Count; idx++)
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

		private double TrainEpoch(int epoch, VMatrix features, VMatrix labels)
		{
			double sse = 0;
			var lo = new object();

			Console.Write("TrainEpoch ");
			var cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				Console.SetCursorPosition(cl, Console.CursorTop);
				Console.Write(row);

				DropNodes();

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
								for (var w = 0; w < node.weights.Length - 1; w++)
								{
									var nNode = m_layers[layer - 1][w];
									if (nNode.isActive)
									{
										node.net += node.weights[w] * nNode.output;
									}
								}
								// add the bias
								node.net += node.weights[node.weights.Length - 1];

								// calculate the output
								node.output = 1.0 / (1.0 + Math.Exp(-node.net));
							}
						}
					});
				}

				// calculate the error and weight changes
				for (var layer = m_layers.Count - 1; layer > 0; layer--)
				{
					Parallel.ForEach(m_layers[layer], node =>
					{
						if (node.isActive)
						{
							var fPrime = node.output * (1.0 - node.output);
							if (layer == m_layers.Count - 1)
							{
								// output layer
								var oNode = node as OutputNode;
								var target = labels.Get(row, oNode.labelCol);
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
							for (var w = 0; w < node.weights.Length - 1; w++)
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
							delta += m_momentum * node.deltas[node.weights.Length - 1];
							node.deltas[node.weights.Length - 1] = delta;
						}
					});
				}

				// update the weights
				for (var layer = 1; layer < m_layers.Count; layer++)
				{
					Parallel.ForEach(m_layers[layer], node =>
					{
						if (node.isActive)
						{
							for (var w = 0; w < node.weights.Length - 1; w++)
							{
								var wNode = m_layers[layer - 1][w];
								if (wNode.isActive)
								{
									node.weights[w] += node.deltas[w];
								}

								// update the bias weight
								node.weights[node.weights.Length - 1] += node.deltas[node.weights.Length - 1];
							}
						}
					});
				}
			}

			Console.WriteLine();

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

			Console.Write("VGetMSE ");
			var cl = Console.CursorLeft;

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
							for (var w = 0; w < node.weights.Length - 1; w++)
							{
								var weight = node.weights[w];
								if (layer == 1)
								{
									weight *= m_pi;
								}
								else
								{
									weight *= m_ph;
								}
								node.net += weight * m_layers[layer - 1][w].output;
							}
							// add the bias
							node.net += node.weights[node.weights.Length - 1];

							node.output = 1.0 / (1.0 + Math.Exp(-node.net));
						}
					});
				}

				// calculate the error of the output layer
				for (var n = 0; n < m_layers[m_layers.Count - 1].Count; n++)
				{
					var node = m_layers[m_layers.Count - 1][n] as OutputNode;
					var target = labels.Get(row, node.labelCol);
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
						for (var w = 0; w < node.weights.Length - 1; w++)
						{
							var weight = node.weights[w];
							if (layer == 1)
							{
								weight *= m_pi;
							}
							else
							{
								weight *= m_ph;
							}
							node.net += weight * m_layers[layer - 1][w].output;
						}
						// add the bias
						node.net += node.weights[node.weights.Length - 1];

						node.output = 1.0 / (1.0 + Math.Exp(-node.net));
					}
				});
			}

			var labelIdx = 0;
			for (var n = 0; n < m_layers[m_layers.Count - 1].Count; n++)
			{
				var node = m_layers[m_layers.Count - 1][n] as OutputNode;

				if (node.isContinuous)
				{
					labels[labelIdx++] = node.output;
				}
				else
				{
					// find the max output for this labelCol
					var max = node.output;
					var labelCol = node.labelCol;
					var labelVal = node.labelVal;
					int nIdx;
					for (nIdx = 1; nIdx + n < m_layers[m_layers.Count - 1].Count; nIdx++)
					{
						var tn = m_layers[m_layers.Count - 1][n + nIdx] as OutputNode;
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
