package com.immomo.nearby.people.conf;

import com.google.common.base.Preconditions;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

import java.io.File;

public final class FMConfigUtil {
    private static Config DEFAULT_CONFIG = null;

    private FMConfigUtil() {
    }

    /**
     * Returns the default {@code Config} object for this app, based on config in the JAR file
     * or otherwise specified to the library.
     */
    public static synchronized Config getDefaultConfig() {
        if (DEFAULT_CONFIG == null) {
            DEFAULT_CONFIG = ConfigFactory.load("FM.conf");
        }
        return DEFAULT_CONFIG;
    }

    /**
     * Loads default configuration including a .conf file as though it were supplied via -Dconfig.file=userConfig
     * <p/>
     * Do NOT use this from user code.  Only to be used in test code.
     */
    public static synchronized void loadUserConfig(String userConfig) {
        if (DEFAULT_CONFIG == null) {
            DEFAULT_CONFIG = ConfigFactory.load(userConfig);
        }
    }

    public static synchronized void overlayConfigOnDefault(File configFile) {
        if (configFile.exists()) {
            Preconditions.checkArgument(!configFile.isDirectory(), "Cannot handle directories of config files %s", configFile);
            DEFAULT_CONFIG = ConfigFactory.parseFileAnySyntax(configFile).resolve().withFallback(getDefaultConfig());
        }
    }

    public static synchronized void overlayConfigOnDefault(String config) {
        if (config != null) {
            DEFAULT_CONFIG = ConfigFactory.parseString(config).resolve().withFallback(getDefaultConfig());
        }
    }
}


