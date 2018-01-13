using System;

namespace MLSystemManager
{
	public class Rand
	{
		private static Random _random;

		private Rand()
		{
			// no instantiate-o
		}

		public static Random Get(int? seed = null)
		{
			return _random ?? (_random = (seed == null ? new Random() : new Random(seed.Value)));
		}
	}
}
