package io.github.orionlibs.orion_llm4j_train;

import io.github.orionlibs.orion_tuple.Pair;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class Tokeniser
{
    protected Map<Pair<Integer, Integer>, Integer> merges;
    private String pattern;
    private Map<String, Integer> specialTokens;
    protected Map<Integer, byte[]> vocabulary;


    public Tokeniser()
    {
        this.merges = new HashMap<>();
        this.pattern = "";
        this.specialTokens = new HashMap<>();
        this.vocabulary = buildVocabulary();
    }


    public abstract void train(String text, int vocabularySize);


    public abstract List<Integer> encode(String text);


    public abstract String decode(List<Integer> tokenIDs);


    //vocabulary is simply and deterministically derived from merges
    private Map<Integer, byte[]> buildVocabulary()
    {
        Map<Integer, byte[]> vocabulary = new HashMap<>();
        //256 is the vocabulary size
        for(int i = 0; i < 256; i++)
        {
            vocabulary.put(i, new byte[] {(byte)i});
        }
        // Process merges
        for(Map.Entry<Pair<Integer, Integer>, Integer> entry : merges.entrySet())
        {
            Pair<Integer, Integer> pair = entry.getKey();
            int p0 = pair.getFirst();
            int p1 = pair.getSecond();
            int idx = entry.getValue();
            // Concatenate byte arrays from vocabulary.get(p0) and vocabulary.get(p1)
            byte[] mergedBytes = new byte[vocabulary.get(p0).length + vocabulary.get(p1).length];
            System.arraycopy(vocabulary.get(p0), 0, mergedBytes, 0, vocabulary.get(p0).length);
            System.arraycopy(vocabulary.get(p1), 0, mergedBytes, vocabulary.get(p0).length, vocabulary.get(p1).length);
            // Add the merged byte array to vocab
            vocabulary.put(idx, mergedBytes);
        }
        // Process special tokens
        for(Map.Entry<String, Integer> entry : specialTokens.entrySet())
        {
            String special = entry.getKey();
            int idx = entry.getValue();
            vocabulary.put(idx, special.getBytes(StandardCharsets.UTF_8));
        }
        return vocabulary;
    }


    public void saveModel(String filePrefix) throws IOException
    {
        // Write the model file
        String modelFile = filePrefix + ".model";
        try(BufferedWriter modelWriter = new BufferedWriter(new FileWriter(modelFile)))
        {
            // Write the version, pattern, and merges
            modelWriter.write("minbpe v1\n");
            modelWriter.write(pattern + "\n");
            // Write special tokens
            modelWriter.write(specialTokens.size() + "\n");
            for(Map.Entry<String, Integer> entry : specialTokens.entrySet())
            {
                modelWriter.write(entry.getKey() + " " + entry.getValue() + "\n");
            }
            // Write the merges
            for(Pair<Integer, Integer> pair : merges.keySet())
            {
                modelWriter.write(pair.getFirst() + " " + pair.getSecond() + "\n");
            }
        }
        // Write the vocabulary file (human-readable)
        String vocabularyFile = filePrefix + ".vocab";
        Map<Integer, Pair<Integer, Integer>> invertedMerges = new HashMap<>();
        for(Map.Entry<Pair<Integer, Integer>, Integer> entry : merges.entrySet())
        {
            invertedMerges.put(entry.getValue(), entry.getKey());
        }
        try(BufferedWriter vocabularyWriter = new BufferedWriter(new FileWriter(vocabularyFile, StandardCharsets.UTF_8)))
        {
            for(Map.Entry<Integer, byte[]> entry : vocabulary.entrySet())
            {
                int idx = entry.getKey();
                byte[] token = entry.getValue();
                String s = Utils.renderToken(token);
                // Check if the token has children (i.e., if it's a merged token)
                if(invertedMerges.containsKey(idx))
                {
                    Pair<Integer, Integer> pair = invertedMerges.get(idx);
                    String s0 = Utils.renderToken(vocabulary.get(pair.getFirst()));
                    String s1 = Utils.renderToken(vocabulary.get(pair.getSecond()));
                    vocabularyWriter.write("[" + s0 + "][" + s1 + "] -> [" + s + "] " + idx + "\n");
                }
                else
                {
                    // Leaf token (first 256 tokens)
                    vocabularyWriter.write("[" + s + "] " + idx + "\n");
                }
            }
        }
    }


    public void loadModel(String modelFile) throws IOException
    {
        if(!modelFile.endsWith(".model"))
        {
            throw new IllegalArgumentException("File must end with '.model'");
        }
        // Initialize the merges and special tokens maps
        Map<Pair<Integer, Integer>, Integer> merges = new HashMap<>();
        Map<String, Integer> specialTokens = new HashMap<>();
        int idx = 256;
        // Read the model file
        try(BufferedReader reader = new BufferedReader(new FileReader(modelFile, StandardCharsets.UTF_8)))
        {
            // Read the version
            String version = reader.readLine().strip();
            if(!"minbpe v1".equals(version))
            {
                throw new IllegalStateException("Unexpected version: " + version);
            }
            // Read the pattern
            this.pattern = reader.readLine().strip();
            // Read the special tokens
            int numSpecial = Integer.parseInt(reader.readLine().strip());
            for(int i = 0; i < numSpecial; i++)
            {
                String[] tokenLine = reader.readLine().strip().split(" ");
                String special = tokenLine[0];
                int specialIdx = Integer.parseInt(tokenLine[1]);
                specialTokens.put(special, specialIdx);
            }
            // Read the merges
            String line;
            while((line = reader.readLine()) != null)
            {
                String[] tokens = line.split(" ");
                int idx1 = Integer.parseInt(tokens[0]);
                int idx2 = Integer.parseInt(tokens[1]);
                merges.put(Pair.of(idx1, idx2), idx);
                idx++;
            }
        }
        // Set the instance fields
        this.merges = merges;
        this.specialTokens = specialTokens;
        // Build the vocab (assuming _build_vocab() is implemented)
        this.vocabulary = buildVocabulary();
    }
}
