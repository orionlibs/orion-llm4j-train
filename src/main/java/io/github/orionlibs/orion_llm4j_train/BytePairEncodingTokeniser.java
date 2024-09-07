package io.github.orionlibs.orion_llm4j_train;

import io.github.orionlibs.orion_assert.Assert;
import io.github.orionlibs.orion_tuple.Pair;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Does not handle the regular expression splitting pattern.
 * Does not handle any special tokens.
 */
public class BytePairEncodingTokeniser extends Tokeniser
{
    public BytePairEncodingTokeniser()
    {
        super();
    }


    @Override
    public void train(String text, int vocabularySize)
    {
        Assert.isGreaterOrEqualTo(vocabularySize, 256, "vocabularySize has to be >= 256");
        int numberOfMergesToDo = vocabularySize - 256;
        //input text preprocessing
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
        //list of integers in range 0..255
        List<Integer> tokenIDs = IntStream.range(0, textBytes.length)
                        .mapToObj(i -> (int)textBytes[i])   // box each byte as Byte
                        .collect(Collectors.toList());
        //iteratively merge the most common pairs to create new tokens
        Map<Pair<Integer, Integer>, Integer> mergesTemp = new HashMap<>();
        Map<Integer, byte[]> vocabularyTemp = new HashMap<>();
        for(int i = 0; i < 256; i++)
        {
            vocabularyTemp.put(i, new byte[] {(byte)i});
        }
        for(int i = 0; i < numberOfMergesToDo; i++)
        {
            //count up the number of times every consecutive pair appears
            Map<Pair<Integer, Integer>, Integer> frequenciesOfConsecutivePairs = Utils.getFrequencyOfConsecutivePairs(tokenIDs);
            //find the pair with the highest count
            Pair<Integer, Integer> pairWithHighestFrequency = frequenciesOfConsecutivePairs.entrySet()
                            .stream()
                            .max(Entry.comparingByValue())
                            .map(Entry::getKey)
                            .orElse(null);
            //mint a new token: assign it the next available id
            int idx = 256 + i;
            //replace all occurrences of pair in ids with idx
            tokenIDs = Utils.merge(tokenIDs, pairWithHighestFrequency, idx);
            //save the merge
            mergesTemp.put(pairWithHighestFrequency, idx);
            byte[] firstArray = vocabularyTemp.get(pairWithHighestFrequency.getFirst());
            byte[] secondArray = vocabularyTemp.get(pairWithHighestFrequency.getSecond());
            byte[] temp = new byte[firstArray.length + secondArray.length];
            System.arraycopy(firstArray, 0, temp, 0, firstArray.length);
            System.arraycopy(secondArray, 0, temp, firstArray.length, secondArray.length);
            vocabularyTemp.put(idx, temp);
        }
        this.merges = mergesTemp;
        this.vocabulary = vocabularyTemp;
    }


    @Override
    public List<Integer> encode(String text)
    {
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
        List<Byte> textBytesToUse = new ArrayList<>();
        for(byte textByte : textBytes)
        {
            if(textByte > 0)
            {
                textBytesToUse.add(textByte);
            }
        }
        textBytes = new byte[textBytesToUse.size()];
        for(int i = 0; i < textBytesToUse.size(); i++)
        {
            textBytes[i] = textBytesToUse.get(i);
        }
        byte[] textBytes2 = new byte[textBytes.length];
        System.arraycopy(textBytes, 0, textBytes2, 0, textBytes2.length);
        //list of integers in range 0..255
        List<Integer> tokenIDs = IntStream.range(0, textBytes.length)
                        .mapToObj(i -> (int)textBytes2[i])   // box each byte as Byte
                        .collect(Collectors.toList());
        while(tokenIDs.size() >= 2)
        {
            Map<Pair<Integer, Integer>, Integer> frequenciesOfConsecutivePairs = Utils.getFrequencyOfConsecutivePairs(tokenIDs);
            //find the pair with the lowest count
            Pair<Integer, Integer> pairWithLowestFrequency = frequenciesOfConsecutivePairs.entrySet()
                            .stream()
                            .min(Entry.comparingByValue())
                            .map(Entry::getKey)
                            .orElse(null);
            //subtle: if there are no more merges available, the key will
            //result in an inf for every single pair, and the min will be
            //just the first pair in the list, arbitrarily
            //we can detect this terminating case by a membership check
            if(!merges.containsKey(pairWithLowestFrequency))
            {
                break;//nothing else can be merged anymore
            }
            //otherwise let's merge the best pair (lowest merge index)
            Integer idx = merges.get(pairWithLowestFrequency);
            tokenIDs = Utils.merge(tokenIDs, pairWithLowestFrequency, idx);
        }
        return tokenIDs;
    }


    @Override
    public String decode(List<Integer> tokenIDs)
    {
        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        for(int id : tokenIDs)
        {
            byte[] vocabBytes = vocabulary.get(id);
            if(vocabBytes != null)
            {
                try
                {
                    byteStream.write(vocabBytes);
                }
                catch(IOException e)
                {
                    // Handle IOException if it occurs
                    e.printStackTrace();
                }
            }
        }
        // Convert the byte array to a string, using UTF-8 with replacement for invalid sequences
        return new String(byteStream.toByteArray(), StandardCharsets.UTF_8);
    }
}
