using System;
using System.IO;
using System.Net;
using System.Net.NetworkInformation;
using System.Runtime.Serialization.Json;
using System.Text;

namespace MLSystemManager
{
    static class ArgParser
	{
	    public static string ParameterFile { get; set; }
	    public static string InputFile { get; set; }
	    public static string OutputFile { get; set; }

		public static void Parse(string[] argv)
		{
			try
			{
                for (var i = 0; i < argv.Length; i++)
				{
					if (argv[i] == "-I")
					{
						InputFile = argv[++i];
					}
					else if (argv[i] == "-O")
					{
						OutputFile = argv[++i];
					}
					else if (argv[i] == "-P")
					{
						ParameterFile = argv[++i];
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
				Console.WriteLine("MLSystemManager -I [inputFile] -O [outputFile] -P [parameterFile]\n");
				Environment.Exit(0);
			}

			if (InputFile == null && ParameterFile == null)
			{
				Console.WriteLine("Usage:");
			    Console.WriteLine("MLSystemManager -I [inputFile] -O [outputFile] -P [parameterFile]\n");
			    Console.WriteLine("-I or -P are required\n");
				Environment.Exit(0);
			}

/*
            //Create a stream to serialize the object to.  
		    var ms = new MemoryStream();

		    // Serializer the User object to the stream.  
		    var ser = new DataContractJsonSerializer(typeof(Parameters));
		    ser.WriteObject(ms, _parameters);
		    byte[] json = ms.ToArray();
		    ms.Close();

            var w = new StreamWriter("Parameters.txt");
		    w.Write(Encoding.UTF8.GetString(json, 0, json.Length));
            w.Close();
*/
        }
	}
}
