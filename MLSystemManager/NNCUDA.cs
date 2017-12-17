using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace MLSystemManager
{
	public class NNCUDA : SupervisedLearner
	{
		private int[] m_lCount;					// number of nodes in each layer
		private int[] m_lBegIdx;				// beginning node index for each layer 
		private Node[] m_nodes;					// node array
		private double[] m_weights;				// weights array
		private double[] m_bestWeights;			// array of best weights
		private double[] m_deltas;				// array of deltas
		private Random m_rand;
		private double m_rate = 0.1;
		private double m_momentum = 0.9;
		private StreamWriter m_outputFile = null;

		[Cudafy]
		public struct Node
		{
			public int wBegIdx;					// beginning index to weights array
			public int wEndIdx;					// end index to weights array
			public int isContinuous;			// true if the outout is continuous
			public int labelCol;				// the label column that this output node corresponds to
			public double labelVal;				// the value of the label column that this output node corresponds to
			public double net;					// the net for this node
			public double output;				// the output of this node
			public double error;				// the error for this node

			public Node(int wBegIdx, int wEndIdx, int isContinuous, int labelCol, double labelVal)
			{
				this.wBegIdx = wBegIdx;
				this.wEndIdx = wEndIdx;
				this.isContinuous = isContinuous;
				this.labelCol = labelCol;
				this.labelVal = labelVal;

				this.net = 0;
				this.output = 0;
				this.error = 0;
			}

		}

		private void SaveBestWeights()
		{
			for (var w = 0; w < m_weights.Length; w++)
			{
				m_bestWeights[w] = m_weights[w];
			}
		}

		private void RestoreBestWeights()
		{
			for (var w = 0; w < m_weights.Length; w++)
			{
				m_weights[w] = m_bestWeights[w];
			}
		}

		private void Shuffle(ref int[] fIdx, Random rand)
		{
			for (int n = fIdx.Length; n > 0; n--)
			{
				int i = rand.Next(n);
				int tmp = fIdx[n - 1];
				fIdx[n - 1] = fIdx[i];
				fIdx[i] = tmp;
			}
		}

		public NNCUDA()
		{
			m_lCount = null;
			m_nodes = null;
			m_weights = null;
			m_bestWeights = null;
			m_deltas = null;
			m_rand = new Random();
		}

		public NNCUDA(Random rand, double rate, double momentum, int[] hidden)
		{
			m_rand = rand;
			m_rate = rate;
			m_momentum = momentum;

			// create the layers index - include input and output
			m_lCount = new int[hidden.Length + 2];
			for (var l = 0; l < hidden.Length; l++)
			{
				m_lCount[l + 1] = hidden[l];
			}
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			if ((m_lCount == null) || (m_lCount.Length < 3))
			{
				m_lCount = new int[3] { 0, features.Cols() * 2, 0 };
			}

			List<Node> nodes = new List<Node>();

			// add the input nodes
			m_lCount[0] = features.Cols();
			for (var n = 0; n < m_lCount[0]; n++)
			{
				nodes.Add(new Node(-1, -1, 0, 0, 0));
			}

			int numWeights = m_lCount[0] + 1;
			int wBegIdx = 0;

			// add the nodes for the hidden layers
			for (var layer = 1; layer < m_lCount.Length - 1; layer++)
			{
				for (var n = 0; n < m_lCount[layer]; n++)
				{
					nodes.Add(new Node(wBegIdx, wBegIdx + numWeights - 1, 0, 0, 0));
					wBegIdx += numWeights;
				}

				numWeights = m_lCount[layer] + 1;
			}

			// figure out how many outputs we need
			int oCount = 0;
			for (var col = 0; col < labels.Cols(); col++)
			{
				var labelValueCount = labels.ValueCount(col);

				if (labelValueCount < 2)
				{
					// continuous
					oCount++;
				}
				else
				{
					oCount += labelValueCount;
				}
			}

			// update the layer arrays
			m_lCount[m_lCount.Length - 1] = oCount;
			m_lBegIdx = new int[m_lCount.Length];
			for (var i = 0; i < m_lCount.Length; i++)
			{
				if (i == 0)
				{
					m_lBegIdx[i] = 0;
				}
				else
				{
					m_lBegIdx[i] = m_lBegIdx[i - 1] + m_lCount[i - 1];
				}
			}

			// add the output nodes
			for (var col = 0; col < labels.Cols(); col++)
			{
				var labelValueCount = labels.ValueCount(col);

				if (labelValueCount < 2)
				{
					// continuous
					nodes.Add(new Node(wBegIdx, wBegIdx + numWeights - 1, 1, col, -1));
					wBegIdx += numWeights;
				}
				else
				{
					for (var n = 0; n < labelValueCount; n++)
					{
						nodes.Add(new Node(wBegIdx, wBegIdx + numWeights - 1, 0, col, n));
						wBegIdx += numWeights;
					}
				}
			}

			m_nodes = nodes.ToArray();

			// create the weights
			m_weights = new double[wBegIdx];
			m_bestWeights = new double[wBegIdx];
			m_deltas = new double[wBegIdx];
			for (var i = 0; i < wBegIdx; i++)
			{
				m_weights[i] = (double)(0.1 - (m_rand.NextDouble() * 0.2));
				m_bestWeights[i] = m_weights[i];
				m_deltas[i] = 0;
			}

			//m_weights[0] = 1.0;
			//m_weights[1] = 0.5;
			//m_weights[2] = 0;
			//m_weights[3] = 1.2;
			//m_weights[4] = 0.5;
			//m_weights[5] = 0.5;
			//m_weights[6] = 0.1;
			//m_weights[7] = -0.8;
			//m_weights[8] = -1.3;

			if (!string.IsNullOrEmpty(OutputFileName))
			{
				m_outputFile = File.AppendText(OutputFileName);
			}

			int trainSize = (int)(0.75 * features.Rows());
			double[,] trainFeatures = new double[trainSize, features.Cols()];
			for (int r = 0; r < trainSize; r++)
			{
				for (int c = 0; c < features.Cols(); c++)
				{
					trainFeatures[r, c] = features.Get(r, c);
				}
			}

			double[,] trainLabels = new double[trainSize, labels.Cols()];
			for (int r = 0; r < trainSize; r++)
			{
				for (int c = 0; c < labels.Cols(); c++)
				{
					trainLabels[r, c] = labels.Get(r, c);
				}
			}

			int[] fIdx = new int[trainSize];
			for (int i = 0; i < fIdx.Length; i++)
			{
				fIdx[i] = i;
			}

			VMatrix validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			VMatrix validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			int epoch = 0;							// current epoch number
			int bestEpoch = 0;						// epoch number of best MSE
			int eCount = 0;							// number of epochs since the best MSE
			bool checkDone = false;					// if true, check to see if we're done
			double bestMSE = double.MaxValue;		// best validation MSE so far
			double bestAccuracy = double.MaxValue;	// best validationa accuracy so far

			Console.WriteLine("Epoch\tMSE (validation)\taccuracy (validation)");
			if (m_outputFile != null)
			{
				m_outputFile.Write("Layers: ");
				for (var l = 0; l < m_lCount.Length - 1; l++)
				{
					m_outputFile.Write(m_lCount[l]);
					m_outputFile.Write('x');
				}
				m_outputFile.WriteLine(m_lCount[m_lCount.Length - 1]);
				m_outputFile.WriteLine("Momentum: " + m_momentum);
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
				m_outputFile.WriteLine("Epoch\tMSE (validation)\taccuracy (validation)");
			}

			CudafyModule km = CudafyTranslator.Cudafy();

			GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
			gpu.LoadModule(km);

			for (; ; )
			{
				// shuffle the training set
				Shuffle(ref fIdx, m_rand);

				double[,] g_trainFeatures = gpu.CopyToDevice(trainFeatures);
				double[,] g_trainLabels = gpu.CopyToDevice(trainLabels);
				int[] g_fIdx = gpu.CopyToDevice(fIdx);
				int[] g_lCount = gpu.CopyToDevice(m_lCount);
				int[] g_lBegIdx = gpu.CopyToDevice(m_lBegIdx);
				Node[] g_nodes = gpu.CopyToDevice(m_nodes);
				double[] g_weights = gpu.CopyToDevice(m_weights);
				double[] g_deltas = gpu.CopyToDevice(m_deltas);

				//// Launch trainSize blocks of 1 thread each
				gpu.Launch(trainSize / 256, 256).TrainEpoch(g_trainFeatures, g_trainLabels, g_fIdx, g_lCount, g_lBegIdx, g_nodes, g_weights, g_deltas, m_rate, m_momentum);

				//// copy the arrays back from the GPU to the CPU
				gpu.CopyFromDevice(g_weights, m_weights);
				gpu.CopyFromDevice(g_deltas, m_deltas);
				gpu.CopyFromDevice(g_fIdx, fIdx);

				// free the memory allocated on the GPU
				gpu.FreeAll();

				//TrainEpoch(trainFeatures, trainLabels, fIdx, m_lCount, m_lBegIdx, m_nodes, ref m_weights, ref m_deltas, m_rate, m_momentum, ref trainMSE);

				// check the MSE after this epoch
				double mse = VGetMSE(validationFeatures, validationLabels);

				// check the validation accuracy after this epoch
				double accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);

				Console.WriteLine(string.Format("{0}-{1}\t{2}\t{3}", epoch, eCount, mse, accuracy));
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine(string.Format("{0}-{1}\t{2}\t{3}", epoch, eCount, mse, accuracy));
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
					bestMSE = mse;
					bestAccuracy = accuracy;
					bestEpoch = epoch;
					SaveBestWeights();
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
			};

			if (m_outputFile != null)
			{
				m_outputFile.WriteLine();
				m_outputFile.WriteLine("Weights");
				PrintWeights();
			}

			if ((bestEpoch > 0) && (bestEpoch != epoch))
			{
				RestoreBestWeights();
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine();
					m_outputFile.WriteLine(string.Format("Best Weights (from Epoch {0}, valMSE={1}, valAcc={2})", bestEpoch, bestMSE, bestAccuracy));
					PrintWeights();
				}
			}

			if (m_outputFile != null)
			{
				m_outputFile.Close();
			}
		}

		[Cudafy]
		public static void TrainEpoch(GThread thread, double[,] features, double[,] labels, int[] fIdx, int[] lCount, int[] lBegIdx, Node[] nodes, double[] weights, double[] deltas, double rate, double momentum)
		{
			object lo = new object();
			unsafe
			{
				int row = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
				if (row < fIdx.Length)
				{
					lock (lo)
					{
						// calculate the output
						for (var layer = 0; layer < lCount.Length; layer++)
						{
							for (var n = 0; n < lCount[layer]; n++)
							{
								fixed (Node* node = &nodes[lBegIdx[layer] + n])
								{
									node->net = 0;
									if (layer == 0)
									{
										// input layer
										node->output = (double)features[fIdx[row], n];
									}
									else
									{
										// calculate the net value
										for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
										{
											node->net += weights[node->wBegIdx + w] * nodes[lBegIdx[layer - 1] + w].output;
										}
										// add the bias
										node->net += weights[node->wEndIdx];

										node->output = 1.0 / (1.0 + Math.Exp(-node->net));
									}
								}
							}
						}

						// calculate the error and weight changes
						for (var layer = lCount.Length - 1; layer > 0; layer--)
						{
							for (var n = 0; n < lCount[layer]; n++)
							{
								fixed (Node* node = &nodes[lBegIdx[layer] + n])
								{
									double fPrime = node->output * (1.0 - node->output);
									if (layer == lCount.Length - 1)
									{
										// output layer
										double target = labels[fIdx[row], node->labelCol];
										if (node->isContinuous == 0)
										{
											// nominal
											if (target == node->labelVal)
											{
												target = 0.9;
											}
											else
											{
												target = 0.1;
											}
										}

										var error = target - node->output;
										node->error = error * fPrime;
									}
									else
									{
										// hidden layer
										double sum = 0;
										for (var tn = 0; tn < lCount[layer + 1]; tn++)
										{
											fixed (Node* tNode = &nodes[lBegIdx[layer + 1] + tn])
											{
												sum += tNode->error * weights[tNode->wBegIdx + n];
											}
										}
										node->error = sum * fPrime;
									}

									// calculate the weight changes
									double delta;
									for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
									{
										delta = rate * node->error * nodes[lBegIdx[layer - 1] + w].output;
										delta += momentum * deltas[node->wBegIdx + w];
										deltas[node->wBegIdx + w] = delta;
									}

									// calculate the bias weight change
									delta = rate * node->error;
									delta += momentum * deltas[node->wEndIdx];
									deltas[node->wEndIdx] = delta;
								}
							}
						}

						// update the weights
						for (var w = 0; w < weights.Length; w++)
						{
							weights[w] += deltas[w];
						}
					}
				}
			}
		}

		private void TrainEpoch(double[,] features, double[,] labels, int[] fIdx, int[] lCount, int[] lBegIdx, Node[] nodes, ref double[] weights, ref double[] deltas, double rate, double momentum, ref double[] sse)
		{
			sse[0] = 0;

			unsafe
			{
				for (var row = 0; row < fIdx.Length; row++)
				{
					// calculate the output
					for (var layer = 0; layer < lCount.Length; layer++)
					{
						for (var n = 0; n < lCount[layer]; n++)
						{
							fixed (Node* node = &nodes[lBegIdx[layer] + n])
							{
								node->net = 0;
								if (layer == 0)
								{
									// input layer
									node->output = features[fIdx[row], n];
								}
								else
								{
									// calculate the net value
									for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
									{
										node->net += weights[node->wBegIdx + w] * nodes[lBegIdx[layer - 1] + w].output;
									}
									// add the bias
									node->net += weights[node->wEndIdx];

									node->output = 1.0 / (1.0 + Math.Exp(-node->net));
								}
							}
						}
					}

					// calculate the error and weight changes
					for (var layer = lCount.Length - 1; layer > 0; layer--)
					{
						for (var n = 0; n < lCount[layer]; n++)
						{
							fixed (Node* node = &nodes[lBegIdx[layer] + n])
							{
								double fPrime = node->output * (1.0 - node->output);
								if (layer == lCount.Length - 1)
								{
									// output layer
									double target = labels[fIdx[row], node->labelCol];
									if (node->isContinuous == 0)
									{
										// nominal
										if (target == node->labelVal)
										{
											target = 0.9;
										}
										else
										{
											target = 0.1;
										}
									}

									var error = target - node->output;
									node->error = error * fPrime;
									sse[0] += error * error;
								}
								else
								{
									// hidden layer
									double sum = 0;
									for (var tn = 0; tn < lCount[layer + 1]; tn++)
									{
										fixed (Node* tNode = &nodes[lBegIdx[layer + 1] + tn])
										{
											sum += tNode->error * weights[tNode->wBegIdx + n];
										}
									}
									node->error = sum * fPrime;
								}

								// calculate the weight changes
								double delta;
								for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
								{
									delta = rate * node->error * nodes[lBegIdx[layer - 1] + w].output;
									delta += momentum * deltas[node->wBegIdx + w];
									deltas[node->wBegIdx + w] = delta;
								}

								// calculate the bias weight change
								delta = rate * node->error;
								delta += momentum * deltas[node->wEndIdx];
								deltas[node->wEndIdx] = delta;
							}
						}
					}

					// update the weights
					for (var w = 0; w < weights.Length; w++)
					{
						weights[w] += deltas[w];
					}
				}
			}

			sse[0] /= fIdx.Length;
		}

		private double TrainEpoch(int epoch, VMatrix features, VMatrix labels)
		{
			double sse = 0;

			Console.Write("TrainEpoch ");
			int cl = Console.CursorLeft;

			unsafe
			{
				for (var row = 0; row < features.Rows(); row++)
				{
					if (((row % 100) == 0) || (row == (features.Rows() - 1)))
					{
						Console.SetCursorPosition(cl, Console.CursorTop);
						Console.Write(row);
					}

					// calculate the output
					for (var layer = 0; layer < m_lCount.Length; layer++)
					{
						for (var n = 0; n < m_lCount[layer]; n++)
						{
							fixed (Node* node = &m_nodes[m_lBegIdx[layer] + n])
							{
								node->net = 0;
								if (layer == 0)
								{
									// input layer
									node->output = features.Get(row, n);
								}
								else
								{
									// calculate the net value
									for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
									{
										node->net += m_weights[node->wBegIdx + w] * m_nodes[m_lBegIdx[layer - 1] + w].output;
									}
									// add the bias
									node->net += m_weights[node->wEndIdx];

									node->output = 1.0 / (1.0 + Math.Exp(-node->net));
								}
							}
						}
					}

					// calculate the error and weight changes
					for (var layer = m_lCount.Length - 1; layer > 0; layer--)
					{
						for (var n = 0; n < m_lCount[layer]; n++)
						{
							fixed (Node* node = &m_nodes[m_lBegIdx[layer] + n])
							{
								double fPrime = node->output * (1.0 - node->output);
								if (layer == m_lCount.Length - 1)
								{
									// output layer
									double target = labels.Get(row, node->labelCol);
									if (node->isContinuous == 0)
									{
										// nominal
										if (target == node->labelVal)
										{
											target = 0.9;
										}
										else
										{
											target = 0.1;
										}
									}

									var error = target - node->output;
									node->error = error * fPrime;
									sse += error * error;
								}
								else
								{
									// hidden layer
									double sum = 0;
									for (var tn = 0; tn < m_lCount[layer + 1]; tn++)
									{
										fixed (Node* tNode = &m_nodes[m_lBegIdx[layer + 1] + tn])
										{
											sum += tNode->error * m_weights[tNode->wBegIdx + n];
										}
									}
									node->error = sum * fPrime;
								}

								// calculate the weight changes
								double delta;
								for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
								{
									delta = m_rate * node->error * m_nodes[m_lBegIdx[layer - 1] + w].output;
									delta += m_momentum * m_deltas[node->wBegIdx + w];
									m_deltas[node->wBegIdx + w] = delta;
								}

								// calculate the bias weight change
								delta = m_rate * node->error;
								delta += m_momentum * m_deltas[node->wEndIdx];
								m_deltas[node->wEndIdx] = delta;
							}
						}
					}

					// update the weights
					for (var w = 0; w < m_weights.Length; w++)
					{
						m_weights[w] += m_deltas[w];
					}
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
			int cl = Console.CursorLeft;

			unsafe
			{
				for (var row = 0; row < features.Rows(); row++)
				{
					if (((row % 10) == 0) || (row == (features.Rows() - 1)))
					{
						Console.SetCursorPosition(cl, Console.CursorTop);
						Console.Write(row);
					}

					// calculate the output
					for (var layer = 0; layer < m_lCount.Length; layer++)
					{
						for (var n = 0; n < m_lCount[layer]; n++)
						{
							fixed (Node* node = &m_nodes[m_lBegIdx[layer] + n])
							{
								node->net = 0;

								// calculate the net value
								if (layer == 0)
								{
									// input layer
									node->output = features.Get(row, n);
								}
								else
								{
									// calculate the net value
									for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
									{
										node->net += m_weights[node->wBegIdx + w] * m_nodes[m_lBegIdx[layer - 1] + w].output;
									}
									// add the bias
									node->net += m_weights[node->wEndIdx];

									node->output = 1.0 / (1.0 + Math.Exp(-node->net));
								}
							}
						}
					}

					// calculate the error of the output layer
					for (var n = 0; n < m_lCount[m_lCount.Length - 1]; n++)
					{
						fixed (Node* node = &m_nodes[m_lBegIdx[m_lCount.Length - 1] + n])
						{
							double target = labels.Get(row, node->labelCol);
							if (node->isContinuous == 0)
							{
								// nominal
								if (target == node->labelVal)
								{
									target = 0.9;
								}
								else
								{
									target = 0.1;
								}
							}
							var error = target - node->output;

							// update the error
							sse += error * error;
						}
					}
				}
			}

			Console.WriteLine();

			return sse / features.Rows();
		}

		private void PrintWeights()
		{
			unsafe
			{
				for (var layer = 1; layer < m_lCount.Length; layer++)
				{
					m_outputFile.WriteLine("Layer " + layer);
					for (var n = 1; n < m_lCount[layer]; n++)
					{
						fixed (Node* node = &m_nodes[m_lBegIdx[layer] + n])
						{
							for (var w = node->wBegIdx; w <= node->wEndIdx; w++)
							{
								m_outputFile.Write(string.Format("{0}\t", m_weights[w]));
							}
							m_outputFile.WriteLine();
						}
					}
				}
			}
			m_outputFile.WriteLine();
		}

		public override void Predict(double[] features, double[] labels)
		{
			unsafe
			{
				for (var layer = 0; layer < m_lCount.Length; layer++)
				{
					for (var n = 0; n < m_lCount[layer]; n++)
					{
						fixed (Node* node = &m_nodes[m_lBegIdx[layer] + n])
						{
							node->net = 0;

							// calculate the net value
							if (layer == 0)
							{
								// input layer
								node->output = features[n];
							}
							else
							{
								// calculate the net value
								for (var w = 0; w < node->wEndIdx - node->wBegIdx; w++)
								{
									node->net += m_weights[node->wBegIdx + w] * m_nodes[m_lBegIdx[layer - 1] + w].output;
								}
								// add the bias
								node->net += m_weights[node->wEndIdx];

								node->output = 1.0 / (1.0 + Math.Exp(-node->net));
							}
						}
					}
				}

				int labelIdx = 0;
				for (var n = 0; n < m_lCount[m_lCount.Length - 1]; n++)
				{
					fixed (Node* node = &m_nodes[m_lBegIdx[m_lCount.Length - 1] + n])
					{
						if (node->isContinuous == 1)
						{
							labels[labelIdx++] = node->output;
						}
						else
						{
							// find the max output for this labelCol
							double max = node->output;
							var labelCol = node->labelCol;
							double labelVal = node->labelVal;
							int nIdx;
							for (nIdx = 1; nIdx + n < m_lCount[m_lCount.Length - 1]; nIdx++)
							{
								fixed (Node* tn = &m_nodes[m_lBegIdx[m_lCount.Length - 1] + n + nIdx])
								{
									if (tn->labelCol != labelCol)
									{
										break;
									}
									else if (tn->output > max)
									{
										max = tn->output;
										labelVal = tn->labelVal;
									}
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
	}
}
