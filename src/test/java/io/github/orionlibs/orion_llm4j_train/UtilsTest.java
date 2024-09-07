package io.github.orionlibs.orion_llm4j_train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.orionlibs.orion_tuple.Pair;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;

@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
public class UtilsTest extends ATest
{
    @Test
    void test_getFrequencyOfConsecutivePairs()
    {
        //[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Map<Pair<Integer, Integer>, Integer> result = Utils.getFrequencyOfConsecutivePairs(Arrays.asList(1, 2, 3, 1, 2));
        assertEquals(3, result.size());
        assertEquals(2, result.get(Pair.of(1, 2)));
        assertEquals(1, result.get(Pair.of(2, 3)));
        assertEquals(1, result.get(Pair.of(3, 1)));
    }


    @Test
    void test_getFrequencyOfConsecutivePairs2()
    {
        //[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Map<Pair<Integer, Integer>, Integer> frequencies = new HashMap<>();
        frequencies.put(Pair.of(1, 2), 1);
        Map<Pair<Integer, Integer>, Integer> result = Utils.getFrequencyOfConsecutivePairs(Arrays.asList(1, 2, 3, 1, 2), frequencies);
        assertEquals(3, result.size());
        assertEquals(3, result.get(Pair.of(1, 2)));
        assertEquals(1, result.get(Pair.of(2, 3)));
        assertEquals(1, result.get(Pair.of(3, 1)));
    }


    @Test
    void test_merge()
    {
        List<Integer> result = Utils.merge(Arrays.asList(1, 2, 3, 1, 2), Pair.of(1, 2), 4);
        assertEquals(Arrays.asList(4, 3, 4), result);
    }
}
