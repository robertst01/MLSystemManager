using System;
using System.Collections.Generic;

namespace MLSystemManager
{
	class ArgParser
	{
		string arff;
		string learner;
		string learnExtra;
		string evaluation;
		string evalExtra;
		string rate = "0.1";
		string momentum = "0.9";
		string ph = "0.5";
		string pi = "0.8";
		string outputs = "1";
		List<int> hidden = new List<int>();
		string outputFileName = string.Empty;
		string weightsFileName = string.Empty;
		bool verbose = false;
		bool normalize = false;
		bool normalizeOutputs = false;
		bool prune = false;
		bool distance = false;
		string k = "1";
		string ignore = string.Empty;
		bool sample = false;
		string corruptLevel = "0";
		string gridSize = "1";
		string iterations = "10000";
		string activation = "sigmoid";
		string actParameter = "0,0,1";
		bool trainAll = false;
		string boost = "1.0";

		//You can add more options for specific learning models if you wish
		public ArgParser(string[] argv)
		{
			try
			{
				for (int i = 0; i < argv.Length; i++)
				{
					if (argv[i].Equals("-A"))
					{
						arff = argv[++i];
					}
					else if (argv[i].Equals("-B"))
					{
						boost = argv[++i];
					}
					else if (argv[i].Equals("-AF"))
					{
						activation = argv[++i];
						if (argv[i].Equals("relu") || argv[i].Equals("vrelu"))
						{
							//expecting ReLU leak level or threshold
							actParameter = argv[++i];
						}
					}
					else if (argv[i].Equals("-C"))
					{
						corruptLevel = argv[++i];
					}
					else if (argv[i].Equals("-D"))
					{
						distance = true;
					}
					else if (argv[i].Equals("-E"))
					{
						evaluation = argv[++i];
						if (argv[i].Equals("static"))
						{
							//expecting a test set name
							evalExtra = argv[++i];
						}
						else if (argv[i].Equals("random"))
						{
							//expecting a double representing the percentage for testing
							//Note stratification is NOT done
							evalExtra = argv[++i];
						}
						else if (argv[i].Equals("cross"))
						{
							//expecting the number of folds
							evalExtra = argv[++i];
						}
						else if (!argv[i].Equals("training"))
						{
							Console.WriteLine("Invalid Evaluation Method: " + argv[i]);
							Environment.Exit(0);
						}
					}
					else if (argv[i].Equals("-F"))
					{
						outputFileName = argv[++i];
					}
					else if (argv[i].Equals("-G"))
					{
						gridSize = argv[++i];
					}
					else if (argv[i].StartsWith("-H"))
					{
						string hlNumStr = argv[i];
						int hlNum = 0;
						if ((hlNumStr.Length < 3) || !int.TryParse(hlNumStr.Substring(2, hlNumStr.Length - 2), out hlNum))
						{
							Console.WriteLine("Invalid Hidden Layer: " + hlNumStr);
							Environment.Exit(0);
						}
						if (hlNum > hidden.Count + 1)
						{
							Console.WriteLine("Hidden Layers must be sequential (-H1, -H2, etc.): " + hlNumStr);
							Environment.Exit(0);
						}
						string hlCountStr = argv[++i];
						int hlCount = 0;
						if (!int.TryParse(hlCountStr, out hlCount))
						{
							Console.WriteLine("Invalid Hidden Layer Count: " + hlCountStr);
							Environment.Exit(0);
						}

						hidden.Add(hlCount);
					}
					else if (argv[i].Equals("-I"))
					{
						ignore = argv[++i];
					}
					else if (argv[i].Equals("-IT"))
					{
						iterations = argv[++i];
					}
					else if (argv[i].Equals("-K"))
					{
						k = argv[++i];
					}
					else if (argv[i].Equals("-L"))
					{
						learner = argv[++i];
						if (learner.Equals("clustering"))
						{
							//expecting the type (k, single, complete)
							learnExtra = argv[++i];
						}
					}
					else if (argv[i].Equals("-M"))
					{
						momentum = argv[++i];
					}
					else if (argv[i].Equals("-N"))
					{
						normalize = true;
					}
					else if (argv[i].Equals("-NO"))
					{
						normalizeOutputs = true;
					}
					else if (argv[i].Equals("-O"))
					{
						outputs = argv[++i];
					}
					else if (argv[i].Equals("-P"))
					{
						prune = true;
					}
					else if (argv[i].Equals("-PH"))
					{
						ph = argv[++i];
					}
					else if (argv[i].Equals("-PI"))
					{
						pi = argv[++i];
					}
					else if (argv[i].Equals("-R"))
					{
						rate = argv[++i];
					}
					else if (argv[i].Equals("-S"))
					{
						sample = true;
					}
					else if (argv[i].Equals("-TA"))
					{
						trainAll = true;
					}
					else if (argv[i].Equals("-V"))
					{
						verbose = true;
					}
					else if (argv[i].Equals("-W"))
					{
						weightsFileName = argv[++i];
					}
					else
					{
						Console.WriteLine("Invalid parameter: " + argv[i]);
						Environment.Exit(0);
					}
				}

			}
			catch (Exception e)
			{
				Console.WriteLine(e.Message);
				Console.WriteLine("Usage:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				Console.WriteLine("OPTIONS:");
				Console.WriteLine("-V Print the confusion matrix and learner accuracy on individual class values\n");

				Console.WriteLine("Possible evaluation methods are:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				Environment.Exit(0);
			}

			if (arff == null || learner == null || evaluation == null)
			{
				Console.WriteLine("Usage:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				Console.WriteLine("OPTIONS:");
				Console.WriteLine("-V Print the confusion matrix and learner accuracy on individual class values");
				Console.WriteLine("-N Use normalized data");
				Console.WriteLine();
				Console.WriteLine("Possible evaluation methods are:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				Environment.Exit(0);
			}
		}

		//The getter methods
		public string GetARFF() { return arff; }
		public string GetLearner() { return learner; }
		public string GetLearnParameter() { return learnExtra; }
		public string GetEvaluation() { return evaluation; }
		public string GetEvalParameter() { return evalExtra; }
		public double GetRate() { return double.Parse(rate); }
		public double GetMomentum() { return double.Parse(momentum); }
		public double GetPH() { return double.Parse(ph); }
		public double GetPI() { return double.Parse(pi); }
		public int[] GetHidden() { return hidden.ToArray(); }
		public int GetOutputs() { return int.Parse(outputs); }
		public string GetOutputFileName() { return outputFileName; }
		public string GetWeightsFileName() { return weightsFileName; }
		public bool GetVerbose() { return verbose; }
		public bool GetNormalize() { return normalize; }
		public bool GetNormalizeOutputs() { return normalizeOutputs; }
		public bool GetPrune() { return prune; }
		public bool GetDistance() { return distance; }
		public int GetK() { return int.Parse(k); }
		public string GetIgnore() { return ignore; }
		public bool GetSample() { return sample; }
		public double GetCorruptLevel() { return double.Parse(corruptLevel); }
		public int GetGridSize() { return int.Parse(gridSize); }
		public int GetIterations() { return int.Parse(iterations); }
		public string GetActivation() { return activation; }
		public string GetActParameter() { return actParameter; }
		public bool GetTrainAll() { return trainAll; }
		public double GetBoost() { return double.Parse(boost); }
	}
}
