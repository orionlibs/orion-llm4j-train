package io.github.orionlibs.project_name.config;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Properties;

/**
 * Properties-based class that holds configuration.
 */
public class OrionConfiguration extends Properties
{
    /**
     * The location of the configuration file that has configuration for the features of this plugin.
     */
    public static final String FEATURE_CONFIGURATION_FILE = "/io/github/orionlibs/project_name/configuration/orion-feature-configuration.prop";


    public void loadFeatureConfiguration(InputStream customConfigStream) throws IOException
    {
        load(customConfigStream);
    }


    public static OrionConfiguration loadFeatureConfiguration(Properties customConfig) throws IOException
    {
        OrionConfiguration featureConfiguration = new OrionConfiguration();
        InputStream defaultConfigStream = OrionConfiguration.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        try
        {
            featureConfiguration.loadDefaultAndCustomConfiguration(defaultConfigStream, customConfig);
            return featureConfiguration;
        }
        catch(IOException e)
        {
            throw new IOException("Could not setup feature configuration for project_name: ", e);
        }
    }


    public static OrionConfiguration loadFeatureConfiguration(String customConfigFile) throws IOException
    {
        OrionConfiguration featureConfiguration = new OrionConfiguration();
        InputStream defaultConfigStream = OrionConfiguration.class.getResourceAsStream(customConfigFile);
        try
        {
            featureConfiguration.loadCustomConfiguration(defaultConfigStream);
            return featureConfiguration;
        }
        catch(IOException e)
        {
            throw new IOException("Could not setup feature configuration for project_name: ", e);
        }
    }


    /**
     * It takes default configuration and nullable custom configuration.
     * For each default configuration property, it registers that one if there is no custom
     * configuration for that property or if customConfig is null. Otherwise it registers the custom one.
     * @param defaultConfiguration
     * @param customConfig
     * @throws IOException if an error occurred when reading from the input stream
     */
    public void loadDefaultAndCustomConfiguration(InputStream defaultConfiguration, Properties customConfig) throws IOException
    {
        Properties allProperties = new Properties();
        for(Map.Entry<Object, Object> prop : System.getProperties().entrySet())
        {
            String key = (String)prop.getKey();
            String value = (String)prop.getValue();
            allProperties.put(key, value);
        }
        if(defaultConfiguration != null)
        {
            allProperties.load(defaultConfiguration);
        }
        putAll(allProperties);
        loadCustomConfiguration(customConfig);
    }


    public void loadCustomConfiguration(InputStream customConfig) throws IOException
    {
        Properties allProperties = new Properties();
        allProperties.load(customConfig);
        putAll(allProperties);
    }


    public void loadCustomConfiguration(Properties customConfig)
    {
        Properties allProperties = new Properties();
        if(customConfig != null)
        {
            for(Map.Entry<Object, Object> prop : customConfig.entrySet())
            {
                String key = (String)prop.getKey();
                String value = (String)prop.getValue();
                allProperties.put(key, value);
            }
        }
        putAll(allProperties);
    }


    /**
     * It converts the configuration this object holds into an InputStream
     * @return an InputStream of the configuration this object holds
     * @throws IOException if writing this property list to the specified output stream throws an IOException
     */
    public InputStream getAsInputStream() throws IOException
    {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        store(output, null);
        return new ByteArrayInputStream(output.toByteArray());
    }


    /**
     * remaps the given key to the given value
     * @param key
     * @param value
     */
    public void updateProp(String key, String value)
    {
        put(key, value);
    }


    /**
     * remaps the given keys to the given values
     * @param customConfig
     */
    public void updateProps(Properties customConfig)
    {
        if(customConfig != null)
        {
            putAll(customConfig);
        }
    }
}
