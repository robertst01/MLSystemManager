using System;
using System.Collections.Generic;

namespace MLSystemManager
{
	static class ArgParser
	{
		public static void Parse(string[] argv)
		{
			var parameters = new Parameters();
			var hidden = new List<int>();

			try
			{
				for (var i = 0; i < argv.Length; i++)
				{
					switch (argv[i])
					{
						case "-A":
							parameters.Arff = argv[++i];
							break;
						case "-B":
							parameters.Boost = double.Parse(argv[++i]);
							break;
						case "-BS":
							parameters.BatchSize = int.Parse(argv[++i]);
							break;
						case "-AF":
							parameters.Activation = argv[++i];
							if ((argv[i] == "relu") || (argv[i] == "vrelu"))
							{
								//expecting ReLU leak level or threshold
								parameters.ActParameter = argv[++i];
							}
							break;
						case "-C":
							parameters.CorruptLevel = double.Parse(argv[++i]);
							break;
						case "-D":
							parameters.Distance = true;
							break;
						case "-E":
							parameters.Evaluation = argv[++i];
							if (argv[i] == "static")
							{
								//expecting a test set name
								parameters.EvalExtra = argv[++i];
							}
							else if (argv[i] == "random")
							{
								//expecting a double representing the percentage for testing
								//Note stratification is NOT done
								parameters.EvalExtra = argv[++i];
							}
							else if (argv[i] == "cross")
							{
								//expecting the number of folds
								parameters.EvalExtra = argv[++i];
							}
							else if (argv[i] == "test")
							{
								//expecting a test set name
								parameters.EvalExtra = argv[++i];
							}
							else if (argv[i] != "training")
							{
								Console.WriteLine("Invalid Evaluation Method: " + argv[i]);
								Environment.Exit(0);
							}
							break;
						case "-G":
							parameters.GridSize = int.Parse(argv[++i]);
							break;
						case "-H":
							var hlNumStr = argv[i];
							var hlNum = 0;
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
							var hlCountStr = argv[++i];
							var hlCount = 0;
							if (!int.TryParse(hlCountStr, out hlCount))
							{
								Console.WriteLine("Invalid Hidden Layer Count: " + hlCountStr);
								Environment.Exit(0);
							}

							hidden.Add(hlCount);
							break;
						case "-I":
							parameters.Ignore = argv[++i];
							break;
						case "-IT":
							parameters.Iterations = int.Parse(argv[++i]);
							break;
						case "-K":
							parameters.K = int.Parse(argv[++i]);
							break;
						case "-L":
							parameters.Learner = argv[++i];
							if (parameters.Learner == "clustering")
							{
								//expecting the type (k, single, complete)
								parameters.LearnExtra = argv[++i];
							}
							break;
						case "-M":
							parameters.Momentum = double.Parse(argv[++i]);
							break;
						case "-N":
							parameters.Normalize = true;
							break;
						case "-NO":
							parameters.NormalizeOutputs = true;
							break;
						case "-O":
							parameters.Outputs = int.Parse(argv[++i]);
							break;
						case "-P":
							parameters = Parameters.Load(argv[++i]);
							break;
						case "-PH":
							parameters.Ph = int.Parse(argv[++i]);
							break;
						case "-PI":
							parameters.Pi = int.Parse(argv[++i]);
							break;
						case "-PR":
							parameters.Prune = true;
							break;
						case "-R":
							parameters.Rate = double.Parse(argv[++i]);
							break;
						case "-S":
							parameters.Sample = true;
							break;
						case "-SF":
							parameters.SnapshotFileName = argv[++i];
							break;
						case "-SI":
							parameters.SnapshotInterval = int.Parse(argv[++i]);
							break;
						case "-TA":
							parameters.TrainAll = true;
							break;
						case "-V":
							parameters.Verbose = true;
							break;
						default:
							Console.WriteLine("Invalid parameter: " + argv[i]);
							Environment.Exit(0);
							break;
					}
				}

				if (hidden.Count > 0)
				{
					parameters.Hidden = hidden.ToArray();
				}

				Parameters.Set(parameters);
			}
			catch (Exception e)
			{
				Console.WriteLine(e.Message);
				Console.WriteLine("Usage:");
				Console.WriteLine(
					"MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				Console.WriteLine("OPTIONS:");
				Console.WriteLine("-V Print the confusion matrix and learner accuracy on individual class values\n");

				Console.WriteLine("Possible evaluation methods are:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				Environment.Exit(0);
			}

			if (parameters.Arff == null || parameters.Learner == null || parameters.Evaluation == null)
			{
				Console.WriteLine("Usage:");
				Console.WriteLine(
					"MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
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

	}
}
