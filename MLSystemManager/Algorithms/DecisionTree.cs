using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	public class DecisionTree : SupervisedLearner
	{
		private bool m_prune;
		private StreamWriter m_outputFile = null;
		private int m_nodeNumber;				// unique node number
		private Node m_tree;					// root node of the resulting tree
		private Matrix m_validationFeatures;
		private Matrix m_validationLabels;
        
		class Node
		{
			public int m_nodeNumber;			// unique node number
			public Node m_parent;				// parent node
			public List<Node> m_children;		// child nodes
			public int m_feature;				// feature to split on
			public int m_val;					// feature value for this branch
			public double m_weight;				// weight for this feature
			public double m_output;				// the resulting class
			public double m_entropy;			// entropy for this node
			public Matrix m_features;			// features for this node
			public Matrix m_labels;				// labels for this node
			public double m_accuracy;			// accuracy after pruning this node
			public double m_outputWeight;		// weight for this output - for predicting

			public Node()
			{
			}

			public Node(int nodeNumber)
			{
				m_nodeNumber = nodeNumber;
			}

			public Node(int nodeNumber, Node parent, int feature, int val, double weight)
			{
				m_nodeNumber = nodeNumber;
				m_parent = parent;
				m_feature = feature;
				m_val = val;
				m_weight = weight;
			}
		}

		public DecisionTree()
		{
			m_prune = false;
		}

		public DecisionTree(bool prune)
		{
			m_prune = prune;
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
			if (!string.IsNullOrEmpty(OutputFileName))
			{
				m_outputFile = File.AppendText(OutputFileName);
			}

			int trainSize = (int)(0.75 * features.Rows());
			if (!m_prune)
			{
				trainSize = features.Rows();
			}
			Matrix trainFeatures = new Matrix(features, 0, 0, trainSize, features.Cols());
			Matrix trainLabels = new Matrix(labels, 0, 0, trainSize, labels.Cols());
			if (m_prune)
			{
				m_validationFeatures = new Matrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
				m_validationLabels = new Matrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());
			}

			FixMissingFeatures(trainFeatures);

			m_nodeNumber = 0;
			m_tree = new Node(++m_nodeNumber);
			m_tree.m_features = trainFeatures;
			m_tree.m_labels = trainLabels;

			CreateSubTree(m_tree);

			if (Verbose)
			{
				PrintTree(m_tree, 0);
			}

			double trainingAccuracy = GetAccuracy(trainFeatures, trainLabels);
			if (m_outputFile != null)
			{
				m_outputFile.WriteLine(string.Format("{0} nodes, {1} levels", GetNodeCount(m_tree), GetMaxDepth(m_tree, 1)));
				m_outputFile.WriteLine(string.Format("Accuracy (training): {0}", trainingAccuracy));
			}

			if (m_prune)
			{
				double validationAccuracy = GetAccuracy(m_validationFeatures, m_validationLabels);
				if (m_outputFile != null)
				{
					m_outputFile.WriteLine(string.Format("Accuracy (validation): {0}", validationAccuracy));
				}

				for (;;)
				{
					var maxNode = PruneNode(m_tree);
					if (maxNode != null)
					{
						if (Verbose && m_outputFile != null)
						{
							m_outputFile.WriteLine(string.Format("MaxAccuracy (pruning): node {0}", maxNode.m_nodeNumber));
						}

						if (maxNode.m_accuracy >= validationAccuracy)
						{
							// prune this node
							maxNode.m_children = null;
							maxNode.m_output = maxNode.m_labels.MostCommonValue(0);
							if (maxNode.m_parent != null)
							{
								maxNode.m_feature = maxNode.m_parent.m_feature;
							}
						}
						else
						{
							break;
						}
					}
					else
					{
						break;
					}
				}

				if (m_outputFile != null)
				{
					m_outputFile.WriteLine(string.Format("{0} nodes, {1} levels", GetNodeCount(m_tree), GetMaxDepth(m_tree, 1)));
				}
				PrintTree(m_tree, 0);
			}

			if (m_outputFile != null)
			{
				m_outputFile.Close();
			}
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
		}

		private void FixMissingFeatures(Matrix features)
		{
			for (var row = 0; row < features.Rows(); row++)
			{
				for (var col = 0; col < features.Cols(); col++)
				{
					var feature = features.Get(row, col);
					if (feature == Matrix.MISSING)
					{
						double val;
						if (features.ValueCount(col) < 2)
						{
							// continuous
							val = features.ColumnMean(col);
						}
						else
						{
							// nominal
							val = features.MostCommonValue(col);
						}
						features.Set(row, col, val);
					}
				}
			}
		}

		private void CreateSubTree(Node root)
		{
			root.m_entropy = Entropy(root.m_labels);

			if (IsPure(root.m_labels))
			{
				root.m_output = root.m_labels.Get(0, 0);
				return;
			}
			
			// get the list of features already used
			List<int> usedFeatures = new List<int>();
			Node p = root.m_parent;
			while (p != null)
			{
				usedFeatures.Add(p.m_feature);
				p = p.m_parent;
			}

			// check to see if we have used all the features
			if (usedFeatures.Count == root.m_features.Cols())
			{
				root.m_output = root.m_labels.MostCommonValue(0);
				return;
			}

			int maxFeature = 0;
			double maxGain = double.MinValue;
			for (var f = 0; f < root.m_features.Cols(); f++)
			{
				if (!usedFeatures.Contains(f))
				{
					if (false)
					{
						// info gain
						double ee = 0;
						for (var val = 0; val < root.m_features.ValueCount(f); val++)
						{
							double sv = 1.0 * root.m_features.Data.Count(d => d[f] == val) / root.m_features.Rows();
							if (sv != 0)
							{
								Matrix newFeatures = null;
								Matrix newLabels = null;
								CopyMatrices(root.m_features, root.m_labels, f, val, out newFeatures, out newLabels);
								var se = Entropy(newLabels);
								ee += sv * se;
							}
						}

						if ((root.m_entropy - ee) > maxGain)
						{
							maxFeature = f;
							maxGain = root.m_entropy - ee;
						}
					}
					else
					{
						// Laplacian
						double lv = 0;
						for (var val = 0; val < root.m_features.ValueCount(f); val++)
						{
							Matrix newFeatures = null;
							Matrix newLabels = null;
							CopyMatrices(root.m_features, root.m_labels, f, val, out newFeatures, out newLabels);
							if (newLabels != null)
							{
								int maxClass = 0;
								// get the class with the most elements
								for (var c = 0; c < newLabels.ValueCount(0); c++)
								{
									int cc = newLabels.Data.Count(d => d[0] == c);
									if (cc > maxClass)
									{
										maxClass = cc;
									}
								}

								lv += 1.0 * newLabels.Rows() / root.m_features.Rows() * (maxClass + 1) / (newLabels.Rows() + newLabels.ValueCount(0));
							}
						}

						if (lv > maxGain)
						{
							maxFeature = f;
							maxGain = lv;
						}
					}
				}
			}

			root.m_feature = maxFeature;

			// add the child nodes
			root.m_children = new List<Node>();
			for (var val = 0; val < root.m_features.ValueCount(maxFeature); val++)
			{
				Matrix newFeatures = null;
				Matrix newLabels = null;
				CopyMatrices(root.m_features, root.m_labels, maxFeature, val, out newFeatures, out newLabels);
				if (newLabels == null)
				{
					Node child = new Node(++m_nodeNumber, root, maxFeature, val, 0);
					child.m_output = m_tree.m_labels.MostCommonValue(0);
					root.m_children.Add(child);
				}
				else
				{
					Node child = new Node(++m_nodeNumber, root, maxFeature, val, 1.0 * newFeatures.Rows() / root.m_features.Rows());
					child.m_features = newFeatures;
					child.m_labels = newLabels;
					
					root.m_children.Add(child);

					CreateSubTree(child);
				}
			}
		}

		private bool IsPure(Matrix labels)
		{
			double val = labels.Get(0, 0);

			for (var row = 1; row < labels.Rows(); row++)
			{
				if (val != labels.Get(row, 0))
				{
					return false;
				}
			}

			return true;
		}

		private void CopyMatrices(Matrix features, Matrix labels, int feature, double val, out Matrix newFeatures, out Matrix newLabels)
		{
			newFeatures = null;
			newLabels = null;

			for (var row = 0; row < features.Rows(); row++)
			{
				if (features.Get(row, feature) == val)
				{
					if (newFeatures == null)
					{
						newFeatures = new Matrix(features, row, 0, 1, features.Cols());
					}
					else
					{
						newFeatures.Add(features, row, 0, 1);
					}
					if (newLabels == null)
					{
						newLabels = new Matrix(labels, row, 0, 1, labels.Cols());
					}
					else
					{
						newLabels.Add(labels, row, 0, 1);
					}
				}
			}
		}

		private double Entropy(Matrix labels)
		{
			double entropy = 0;

			for (var val = 0; val < labels.ValueCount(0); val++)
			{
				double p = 1.0 * labels.Data.Count(d => d[labels.Cols() - 1] == val) / labels.Rows();
				if (p != 0)
				{
					entropy += -(p * Math.Log10(p) / Math.Log10(2));
				}
			}

			return entropy;
		}

		private Node PruneNode(Node node)
		{
			if (node.m_children == null)
			{
				return null;
			}

			double maxAccuracy = 0;
			Node maxNode = null;
			foreach (var c in node.m_children)
			{
				Node n = PruneNode(c);
				if ((n != null) && n.m_accuracy > maxAccuracy)
				{
					maxAccuracy = n.m_accuracy;
					maxNode = n;
				}
			}

			// prune this node and check accuracy
			List<Node> children = node.m_children;
			node.m_children = null;
			double output = node.m_output;
			node.m_output = node.m_labels.MostCommonValue(0);
			int feature = node.m_feature;
			if (node.m_parent != null)
			{
				node.m_feature = node.m_parent.m_feature;
			}
			node.m_accuracy = GetAccuracy(m_validationFeatures, m_validationLabels);
			if (Verbose && m_outputFile != null)
			{
				m_outputFile.WriteLine(string.Format("Pruning {0}\taccuracy: {1}", node.m_nodeNumber, node.m_accuracy));
			}

			node.m_children = children;
			node.m_output = output;
			node.m_feature = feature;

			if (node.m_accuracy > maxAccuracy)
			{
				return node;
			}
			else
			{
				return maxNode;
			}
		}

		private int GetNodeCount(Node node)
		{
			if (node.m_children == null)
			{
				return 1;
			}
			else
			{
				int count = 1;
				foreach (var c in node.m_children)
				{
					count += GetNodeCount(c);
				}

				return count;
			}
		}

		private int GetMaxDepth(Node node, int depth)
		{
			if (node.m_children == null)
			{
				return depth;
			}
			else
			{
				int max = 0;
				foreach (var c in node.m_children)
				{
					int m = GetMaxDepth(c, depth + 1);
					if (m > max)
					{
						max = m;
					}
				}

				return max;
			}
		}

		private void PrintTree(Node node, int level)
		{
			if (m_outputFile != null)
			{
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Node: {0}, Level: {1}", node.m_nodeNumber, level));
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Parent: {0}", node.m_parent == null ? "null" : node.m_parent.m_nodeNumber.ToString()));
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Feature: {0}", node.m_feature));
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Val: {0}", node.m_val));
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Weight: {0}", node.m_weight));
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Output: {0}", node.m_output));
				PrintTabs(level);
				m_outputFile.WriteLine(string.Format("Entropy: {0}", node.m_entropy));

				if (node.m_children != null)
				{
					PrintTabs(level);
					m_outputFile.Write("Children: ");
					for (var c = 0; c < node.m_children.Count; c++)
					{
						if (c > 0)
						{
							m_outputFile.Write(", ");
						}
						m_outputFile.Write(node.m_children[c].m_nodeNumber);
					}
					m_outputFile.WriteLine();
				}

				m_outputFile.WriteLine();

				if (node.m_children != null)
				{
					foreach (var n in node.m_children)
					{
						PrintTree(n, level + 1);
					}
				}
			}
		}

		private void PrintTabs(int count)
		{
			for (var i = 0; i < count; i++)
			{
				m_outputFile.Write("\t");
			}
		}

		private double GetAccuracy(Matrix features, Matrix labels)
		{
			int correct = 0;

			for (var row = 0; row < features.Rows(); row++)
			{
				var output = GetOutput(features.Row(row));
				if (output == labels.Get(row, 0))
				{
					correct++;
				}
			}

			return 1.0 * correct / features.Rows();
		}

		private double GetOutput(double[] features)
		{
			int max = ClearOutputs(m_tree);
			SetOutputs(features, m_tree, 1.0);
			var outputs = new double[max + 1];
			GetOutputs(m_tree, outputs);

			double maxVal = double.MinValue;
			int maxOutput = 0;
			for (var i = 0; i < outputs.Length; i++)
			{
				if (outputs[i] > maxVal)
				{
					maxVal = outputs[i];
					maxOutput = i;
				}
			}
			
			return maxOutput;
		}

		private int ClearOutputs(Node node)
		{
			node.m_outputWeight = 0;
			if (node.m_children == null)
			{
				return (int)node.m_output;
			}
			else
			{
				int max = 0;
				foreach (var c in node.m_children)
				{
					int m = ClearOutputs(c);
					if (m > max)
					{
						max = m;
					}
				}

				return max;
			}
		}

		private void SetOutputs(double[] features, Node node, double weight)
		{
			if (node.m_children == null)
			{
				node.m_outputWeight = weight;
			}
			else
			{
				double val = features[node.m_feature];
				if (val != Matrix.MISSING)
				{
					node = node.m_children.Where(c => c.m_val == val).FirstOrDefault();
					if (node == null)
					{
						throw new Exception("Invalid value in features");
					}
					else
					{
						SetOutputs(features, node, weight);
					}
				}
				else
				{
					foreach (var c in node.m_children)
					{
						SetOutputs(features, c, weight * c.m_weight);
					}
				}
			}
		}

		private void GetOutputs(Node node, double[] outputs)
		{
			if (node.m_children == null)
			{
				outputs[(int)node.m_output] += node.m_outputWeight;
			}
			else
			{
				foreach (var c in node.m_children)
				{
					GetOutputs(c, outputs);
				}
			}
		}

		public override void Predict(double[] features, double[] labels)
		{
			labels[0] = GetOutput(features);
		}
	}
}
