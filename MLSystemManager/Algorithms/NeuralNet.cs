#define parallel

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	public class NeuralNet : SupervisedLearner
	{
		public List<Layer> Layers { get; set; }
		private readonly Random _rand;

		public enum LayerType
		{
			Input,
			Hidden,
			Output
		}

		public class Layer
		{
			public LayerType Type { get; set; }
			public List<Node> Nodes { get; set; }
			public Layer Previous { get; set; }
			public Layer Next { get; set; }
		}

		public class Node
		{
			public int Index { get; set; }					// the index of this node
			public double Net { get; set; }					// the net for this node
			public double Output { get; set; }				// the output of this node
			public double Error { get; set; }				// the error for this node
			public double[] Weights { get; set; }			// the weights for all nodes connected to this node
			public double[] BestWeights { get; set; }		// the best weights so far
			public double[] Deltas { get; set; }            // weight deltas for current epoch
			public double[] PrevDeltas { get; set; }        // weight deltas from previous epoch

			public Node()
			{
				Index = 0;
				Net = 0;
				Output = 0;
				Error = 0;
				Weights = null;
				BestWeights = null;
				Deltas = null;
				PrevDeltas = null;
			}

			public Node(int idx, int numWeights, int numNodes, Random rand)
			{
				Index = idx;
				if (numWeights > 0)
				{
					// see Bengio, X. Glorot, Understanding the difficulty of training deep feedforward neuralnetworks, AISTATS 2010
					var initMax = 4.0 * Math.Sqrt(6.0 / (numWeights - 1 + numNodes));

					Weights = new double[numWeights];
					BestWeights = new double[numWeights];
					for (var i = 0; i < numWeights; i++)
					{
						Weights[i] = initMax - (rand.NextDouble() * initMax * 2.0);
						BestWeights[i] = Weights[i];
					}

					Deltas = new double[numWeights];
					PrevDeltas = new double[numWeights];
					for (var i = 0; i < numWeights; i++)
					{
						Deltas[i] = 0;
						PrevDeltas[i] = 0;
					}
				}
			}

			public Node(int idx, double[] weights)
			{
				Index = idx;
				Weights = weights;
				BestWeights = new double[weights.Length];
				for (var i = 0; i < weights.Length; i++)
				{
					BestWeights[i] = Weights[i];
				}

				Deltas = new double[weights.Length];
				PrevDeltas = new double[weights.Length];
				for (var i = 0; i < weights.Length; i++)
				{
					Deltas[i] = 0;
					PrevDeltas[i] = 0;
				}
			}

			public void SaveBestWeights()
			{
				if (Weights != null)
				{
					for (var i = 0; i < Weights.Length; i++)
					{
						BestWeights[i] = Weights[i];
					}
				}
			}

			public void RestoreBestWeights()
			{
				if (Weights != null)
				{
					for (var i = 0; i < Weights.Length; i++)
					{
						Weights[i] = BestWeights[i];
					}
				}
			}
		}

		public class InputNode : Node
		{
			public int Feature { get; set; }
			public InputNode(int idx, int feature, Random rand)
				: base(idx, 0, 0, rand)
			{
				Feature = feature;
			}
		}

		public class HiddenNode : Node
		{
			public HiddenNode(int idx, int numWeights, int numNodes, Random rand)
				: base(idx, numWeights, numNodes, rand)
			{
			}
			public HiddenNode(int idx, double[] weights)
				: base(idx, weights)
			{
			}
		}

		public class OutputNode : Node
		{
			public bool IsContinuous { get; set; }			// true if the outout is continuous
			public int LabelCol { get; set; }				// the label column that this output node corresponds to
			public double LabelVal { get; set; }			// the value of the label column that this output node corresponds to
			public OutputNode(int idx, bool isContinuous, int labelCol, double labelVal, int numWeights, int numNodes, Random rand)
				: base(idx, numWeights, numNodes, rand)
			{
				IsContinuous = isContinuous;
				LabelCol = labelCol;
				LabelVal = labelVal;
			}
			public OutputNode(int idx, bool isContinuous, int labelCol, double labelVal, double[] weights)
				: base(idx, weights)
			{
				IsContinuous = isContinuous;
				LabelCol = labelCol;
				LabelVal = labelVal;
			}
		}

		public NeuralNet()
		{
			_rand = new Random();
			Layers = new List<Layer>();
		}

		public NeuralNet(Parameters parameters)
		{
			_rand = Rand.Get();
			Parameters = parameters;
			Layers = new List<Layer>();

			if (!string.IsNullOrEmpty(parameters.SnapshotFileName) && File.Exists(parameters.SnapshotFileName))
			{
				 NeuralNetSave.Load(parameters.SnapshotFileName, this);
			}
		}

		public override void Train(Matrix features, Matrix labels)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels)
		{
			if (Layers.Count < 1)
			{
				// create the layers
				if (Parameters.Hidden.Length < 1)
				{
					Parameters.Hidden = new[] {features.Cols() * 2};
				}

				// add the input nodes
				var iNodes = new List<Node>();
				for (var i = 0; i < features.Cols(); i++)
				{
					iNodes.Add(new InputNode(i, i, _rand));
				}

				var iLayer = new Layer()
				{
					Type = LayerType.Input,
					Nodes = iNodes,
					Previous = null,
					Next = null
				};
				Layers.Add(iLayer);

				var prevLayer = iLayer;

				// add the hidden nodes
				foreach (var t in Parameters.Hidden)
				{
					var hNodes = new List<Node>();

					for (var n = 0; n < t; n++)
					{
						var node = new HiddenNode(n, prevLayer.Nodes.Count + 1, t, _rand);
						hNodes.Add(node);
					}

					var hLayer = new Layer()
					{
						Type = LayerType.Hidden,
						Nodes = hNodes,
						Previous = prevLayer,
						Next = null
					};
					Layers.Add(hLayer);

					prevLayer.Next = hLayer;
					prevLayer = hLayer;
				}

				// add the output nodes
				var oNodes = new List<Node>();
				var oCount = 0;
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

				for (var col = 0; col < labels.Cols(); col++)
				{
					var labelValueCount = labels.ValueCount(col);

					if (labelValueCount < 2)
					{
						// continuous
						var node = new OutputNode(oNodes.Count, true, col, -1, prevLayer.Nodes.Count + 1, oCount, _rand);
						oNodes.Add(node);
					}
					else
					{
						for (var n = 0; n < labelValueCount; n++)
						{
							var node = new OutputNode(oNodes.Count, false, col, n, prevLayer.Nodes.Count + 1, oCount, _rand);
							oNodes.Add(node);
						}
					}
				}

				var oLayer = new Layer()
				{
					Type = LayerType.Output,
					Nodes = oNodes,
					Previous = prevLayer
				};
				Layers.Add(oLayer);

				prevLayer.Next = oLayer;
			}

			var trainSize = (int) (0.75 * features.Rows());
			var trainFeatures = new VMatrix(features, 0, 0, trainSize, features.Cols());
			var trainLabels = new VMatrix(labels, 0, 0, trainSize, labels.Cols());
			var validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			var validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			Console.Write("Layers: ");
			foreach (var layer in Layers)
			{
				Console.Write(layer.Nodes.Count);
				if (layer.Type == LayerType.Output)
				{
					Console.WriteLine();
				}
				else
				{
					Console.Write('x');
				}
			}

			Console.WriteLine("AF: " + Parameters.Activation);

			Console.WriteLine("Epoch\tMSE (validation)");

			int epoch; // current epoch number
			var bestEpoch = 0; // epoch number of best MSE
			var eCount = 0; // number of epochs since the best MSE
			var checkDone = false; // if true, check to see if we're done
			var initialMse = Parameters.InitialMse; // MSE for first epoch
			var bestMse = Parameters.StartMse; // best validation MSE so far
			double bestAccuracy = 0;
			var batchCount = (trainFeatures.Rows() + Parameters.BatchSize - 1) / Parameters.BatchSize;
			int countInterval = batchCount / 10;
			if (countInterval < 1)
			{
				countInterval = 1;
			}
			var startEpoch = Parameters.StartEpoch + 1;

			for (epoch = startEpoch;; epoch++)
			{
				// shuffle the training set
				trainFeatures.Shuffle(_rand, trainLabels);
				var cl = Console.CursorLeft;

				for (var batch = 0; batch < batchCount; batch++)
				{
					var startIdx = batch * Parameters.BatchSize;
					var count = Parameters.BatchSize;
					if ((startIdx + count) > trainFeatures.Rows())
					{
						count = trainFeatures.Rows() - startIdx;
					}
					TrainBatch(trainFeatures, trainLabels, startIdx, count);

					if ((((batch + 1) % countInterval) == 0) || (batch == (batchCount - 1)))
					{
						Console.SetCursorPosition(cl, Console.CursorTop);
						Console.Write(batch + 1);
					}
				}

				Console.WriteLine();

				// check the MSE
				var mse = VGetMSE(validationFeatures, validationLabels);
				if ((epoch == startEpoch) && (initialMse == 0))
				{
					// save the initial MSE
					initialMse = mse;
				}
				var accuracy = VMeasureAccuracy(validationFeatures, validationLabels, null);

				if ((epoch % Parameters.SnapshotInterval) == 0)
				{
					SaveSnapshot(epoch, mse, initialMse, accuracy);
				}

				Console.WriteLine($"{epoch}-{eCount}\t{mse}");

				if ((mse == 0) || (epoch > 5000))
				{
					break;
				}

				if ((epoch == startEpoch) || (mse < bestMse))
				{
					if ((epoch != startEpoch) && !checkDone && (mse < initialMse * 0.9))
					{
						checkDone = true;
					}
					eCount = 0;

					// save the best for later
					bestMse = mse;
					bestEpoch = epoch;
					bestAccuracy = accuracy;
					foreach (var layer in Layers)
					{
						foreach (var node in layer.Nodes)
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

			if ((bestEpoch > 0) && (bestEpoch != epoch))
			{
				foreach (var layer in Layers)
				{
					foreach (var node in layer.Nodes)
					{
						node.RestoreBestWeights();
					}
				}
			}

			SaveSnapshot(bestEpoch, bestMse, initialMse, bestAccuracy, true);
		}

		private void SaveSnapshot(int epoch, double mse, double initialMse, double accuracy, bool isFinal = false)
		{
			var final = isFinal ? "final_" : string.Empty;
			var fileName = $"{Path.GetDirectoryName(Parameters.Arff)}{Path.GetFileNameWithoutExtension(Parameters.Arff)}_iter_{final}{epoch}.txt";
			NeuralNetSave.Save(fileName, this, epoch, mse, initialMse, accuracy);
		}

		private void TrainBatch(VMatrix features, VMatrix labels, int startIdx, int count)
		{
			for (var idx = 0; idx < count; idx++)
			{
				var row = startIdx + idx;
				if (row > (features.Rows() - 1))
				{
					row = features.Rows() - 1;
				}

				// calculate the output
				foreach (var layer in Layers)
				{
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						node.Net = 0;
						node.Output = 0;
						node.Error = 0;

						if (layer.Type == LayerType.Input)
						{
							// input node
							node.Net = features.Get(row, node.Index);
							node.Output = node.Net;
						}
						else
						{
							// calculate the net value
							for (var w = 0; w < node.Weights.Length - 1; w++)
							{
								node.Net += node.Weights[w] * layer.Previous.Nodes[w].Output;
							}
							// add the bias
							node.Net += node.Weights[node.Weights.Length - 1];

							// calculate the output
							switch (Parameters.Activation)
							{
								case "relu":
									node.Output = node.Net < 0 ? 0.01 * node.Net : node.Net;
									break;
								case "softsign":
									node.Output = (node.Net / (1.0 + Math.Abs(node.Net)));
									break;
								case "softplus":
									node.Output = Math.Log(1.0 + Math.Exp(node.Net));
									break;
								default:
									node.Output = 1.0 / (1.0 + Math.Exp(-node.Net));
									break;
							}
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error and weight changes
				for (var i = Layers.Count - 1; i > 0; i--)
				{
					var layer = Layers[i];
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						double fPrime;
						switch (Parameters.Activation)
						{
							case "relu":
								fPrime = (node.Output < 0 ? 0.01 : 1);
								break;
							case "softsign":
								fPrime = 1.0 / ((1.0 - Math.Abs(node.Output)) * (1.0 - Math.Abs(node.Output)));
								break;
							case "softplus":
								fPrime = 1.0 / (1.0 + Math.Exp(-node.Output));
								break;
							default:
								fPrime = node.Output * (1.0 - node.Output);
								break;
						}

						if (layer.Type == LayerType.Output)
						{
							// output layer
							var oNode = node as OutputNode;
							var target = labels.Get(row, oNode.LabelCol);
							if (!oNode.IsContinuous)
							{
								// nominal
								target = target == oNode.LabelVal ? 0.99 : 0.01;
							}

							var error = target - node.Output;
							node.Error = error * fPrime;
						}
						else
						{
							// hidden layer
							double sum = 0;
							foreach (var tn in layer.Next.Nodes)
							{
								sum += tn.Error * tn.Weights[node.Index];
							}
							node.Error = sum * fPrime;
						}

						// calculate the weight changes
						double delta;
						for (var w = 0; w < node.Weights.Length - 1; w++)
						{
							delta = Parameters.Rate * node.Error * layer.Previous.Nodes[w].Output;
							delta += Parameters.Momentum * node.PrevDeltas[w];
							node.Deltas[w] += delta;
						}

						// calculate the bias weight change
						delta = Parameters.Rate * node.Error;
						delta += Parameters.Momentum * node.PrevDeltas[node.Weights.Length - 1];
						node.Deltas[node.Weights.Length - 1] += delta;
#if parallel
					});
#else
					}
#endif
				}
			}

			// update the weights
			foreach (var layer in Layers)
			{
				if (layer.Type != LayerType.Input)
				{
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						for (var w = 0; w < node.Weights.Length; w++)
						{
							node.PrevDeltas[w] = node.Deltas[w] / count;
							node.Weights[w] += node.PrevDeltas[w];
							node.Deltas[w] = 0;
						}
#if parallel
					});
#else
					}
#endif
				}
			}
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
				if (((row % 10) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

				// calculate the output
				foreach (var layer in Layers)
				{
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						node.Net = 0;
						node.Output = 0;

						if (layer.Type == LayerType.Input)
						{
							// input node
							node.Output = features.Get(row, node.Index);
						}
						else
						{
							// calculate the net value
							for (var w = 0; w < node.Weights.Length - 1; w++)
							{
								node.Net += node.Weights[w] * layer.Previous.Nodes[w].Output;
							}
							// add the bias
							node.Net += node.Weights[node.Weights.Length - 1];

							// calculate the output
							switch (Parameters.Activation)
							{
								case "relu":
									node.Output = node.Net < 0 ? 0.01 * node.Net : node.Net;
									break;
								case "softsign":
									node.Output = (node.Net / (1.0 + Math.Abs(node.Net)));
									break;
								case "softplus":
									node.Output = Math.Log(1.0 + Math.Exp(node.Net));
									break;
								default:
									node.Output = 1.0 / (1.0 + Math.Exp(-node.Net));
									break;
							}
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error of the output layer
				foreach (var node in Layers[Layers.Count - 1].Nodes)
				{
					var oNode = node as OutputNode;
					var target = labels.Get(row, oNode.LabelCol);
					if (!oNode.IsContinuous)
					{
						// nominal
						if (target == oNode.LabelVal)
						{
							target = 0.99;
						}
						else
						{
							target = 0.01;
						}
					}
					var error = target - node.Output;

					// update the error
					sse += error * error;
				}
			}

			Console.WriteLine();

			return sse / features.Rows();
		}

		public override void Predict(double[] features, double[] labels)
		{
			foreach (var layer in Layers)
			{
#if parallel
				Parallel.ForEach(layer.Nodes, node =>
#else
				foreach (var node in layer.Nodes)
#endif
				{
					node.Net = 0;
					node.Output = 0;

					if (layer.Type == LayerType.Input)
					{
						// input node
						node.Output = features[node.Index];
					}
					else
					{
						// calculate the net value
						for (var w = 0; w < node.Weights.Length - 1; w++)
						{
							node.Net += node.Weights[w] * layer.Previous.Nodes[w].Output;
						}
						// add the bias
						node.Net += node.Weights[node.Weights.Length - 1];

						switch (Parameters.Activation)
						{
							case "relu":
								node.Output = node.Net < 0 ? 0.01 * node.Net : node.Net;
								break;
							case "softsign":
								node.Output = (node.Net / (1.0 + Math.Abs(node.Net)));
								break;
							case "softplus":
								node.Output = Math.Log(1.0 + Math.Exp(node.Net));
								break;
							default:
								node.Output = 1.0 / (1.0 + Math.Exp(-node.Net));
								break;
						}
					}
#if parallel
				});
#else
				}
#endif
			}

			var labelIdx = 0;
			var outputLayer = Layers[Layers.Count - 1];
			for (var n = 0; n < outputLayer.Nodes.Count; n++)
			{
				var oNode = outputLayer.Nodes[n] as OutputNode;

				if (oNode.IsContinuous)
				{
					labels[labelIdx++] = oNode.Output;
				}
				else
				{
					// find the max output for this labelCol
					var max = oNode.Output;
					var labelCol = oNode.LabelCol;
					var labelVal = oNode.LabelVal;
					int nIdx;
					for (nIdx = 1; nIdx + n < outputLayer.Nodes.Count; nIdx++)
					{
						var tn = outputLayer.Nodes[n + nIdx] as OutputNode;
						if (tn.LabelCol != labelCol)
						{
							break;
						}

						if (tn.Output > max)
						{
							max = tn.Output;
							labelVal = tn.LabelVal;
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
