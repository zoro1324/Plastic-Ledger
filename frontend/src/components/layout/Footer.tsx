import React from 'react';
import { motion } from 'framer-motion';
import { Github, Linkedin, Mail, ExternalLink } from 'lucide-react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = [
    {
      section: 'Project',
      links: [
        { label: 'GitHub', href: '#', icon: Github },
        { label: 'Documentation', href: '#' },
      ],
    },
    {
      section: 'Legal',
      links: [
        { label: 'Privacy Policy', href: '#' },
        { label: 'Terms of Service', href: '#' },
      ],
    },
    {
      section: 'Contact',
      links: [
        { label: 'Email', href: 'mailto:hello@plasticledger.ai', icon: Mail },
        { label: 'LinkedIn', href: '#', icon: Linkedin },
      ],
    },
  ];

  return (
    <motion.footer
      className="bg-gradient-to-t from-[#071A2E] to-[#0F2D4A] border-t border-[#0A84FF]/20 text-gray-300"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-br from-[#0A84FF] to-[#00C2A8] rounded-lg flex items-center justify-center text-white font-bold">
                ◆
              </div>
              <span className="text-lg font-bold">PlasticLedger AI</span>
            </div>
            <p className="text-sm text-gray-400">
              AI-powered environmental intelligence platform
            </p>
          </div>

          {/* Footer Links */}
          {footerLinks.map((section, idx) => (
            <motion.div key={idx} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h4 className="font-semibold mb-4 text-white">{section.section}</h4>
              <ul className="space-y-2">
                {section.links.map((link, linkIdx) => (
                  <li key={linkIdx}>
                    <a
                      href={link.href}
                      className="text-sm hover:text-[#4CC9F0] transition-colors flex items-center space-x-1"
                    >
                      {link.icon && <link.icon size={16} />}
                      <span>{link.label}</span>
                      {link.href.startsWith('http') && <ExternalLink size={12} />}
                    </a>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        {/* Divider */}
        <div className="border-t border-[#0A84FF]/20 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm">
            <p>&copy; {currentYear} PlasticLedger AI. All rights reserved.</p>
            <p className="mt-4 md:mt-0">Protecting our oceans with AI and Earth observation</p>
          </div>
        </div>
      </div>
    </motion.footer>
  );
};
