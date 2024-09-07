package io.github.orionlibs.orion_llm4j_train;

import io.github.orionlibs.orion_string.StringsService;
import io.github.orionlibs.orion_tuple.Pair;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Utils
{
    public static Map<Pair<Integer, Integer>, Integer> getFrequencyOfConsecutivePairs(List<Integer> list)
    {
        return getFrequencyOfConsecutivePairs(list, new HashMap<>());
    }


    public static Map<Pair<Integer, Integer>, Integer> getFrequencyOfConsecutivePairs(List<Integer> list, Map<Pair<Integer, Integer>, Integer> frequencies)
    {
        Map<Pair<Integer, Integer>, Integer> frequenciesTemp = frequencies == null ? new HashMap<>() : frequencies;
        for(int i = 0; i < list.size() - 1; i++)
        {
            Pair<Integer, Integer> pair = Pair.of(list.get(i), list.get(i + 1));
            Integer frequency = frequenciesTemp.get(pair);
            if(frequency != null)
            {
                frequenciesTemp.put(pair, frequency + 1);
            }
            else
            {
                frequenciesTemp.put(pair, 1);
            }
        }
        return frequenciesTemp;
    }


    /**
     * replaces occurences of given pair with an integer
     * @param list
     * @param pairToReplace
     * @param replacement
     * @return
     */
    public static List<Integer> merge(List<Integer> list, Pair<Integer, Integer> pairToReplace, int replacement)
    {
        List<Integer> result = new ArrayList<>();
        int i = 0;
        while(i < list.size())
        {
            if(list.get(i).equals(pairToReplace.getFirst())
                            && i < list.size() - 1
                            && list.get(i + 1).equals(pairToReplace.getSecond()))
            {
                result.add(replacement);
                i += 2;
            }
            else
            {
                result.add(list.get(i));
                i += 1;
            }
        }
        return result;
    }


    public static String renderToken(byte[] bytes)
    {
        return StringsService.replaceControlCharacters(new String(bytes, Charset.forName("utf-8")));
    }
}
